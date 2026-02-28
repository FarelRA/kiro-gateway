# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Authentication manager for Kiro API.

Manages the lifecycle of access tokens:
- Loading credentials from .env or JSON file
- Automatic token refresh on expiration
- Thread-safe refresh using asyncio.Lock
- Support for both Kiro Desktop Auth and AWS SSO OIDC (kiro-cli)
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from kiro.config import (
    TOKEN_REFRESH_THRESHOLD,
    get_kiro_refresh_url,
    get_kiro_api_host,
    get_kiro_q_host,
    get_aws_sso_oidc_url,
    KIRO_CLI_DB_AUTO_CLEANUP,
)
from kiro.utils import get_machine_fingerprint


# Supported SQLite token keys (searched in priority order)
SQLITE_TOKEN_KEYS = [
    "kirocli:social:token",      # Social login (Google, GitHub, Microsoft, etc.)
    "kirocli:odic:token",        # AWS SSO OIDC (kiro-cli corporate)
    "codewhisperer:odic:token",  # Legacy AWS SSO OIDC
]

# Device registration keys (for AWS SSO OIDC only)
SQLITE_REGISTRATION_KEYS = [
    "kirocli:odic:device-registration",
    "codewhisperer:odic:device-registration",
]


class AuthType(Enum):
    """
    Type of authentication mechanism.
    
    KIRO_DESKTOP: Kiro IDE credentials (default)
        - Uses https://prod.{region}.auth.desktop.kiro.dev/refreshToken
        - JSON body: {"refreshToken": "..."}
    
    AWS_SSO_OIDC: AWS SSO credentials from kiro-cli
        - Uses https://oidc.{region}.amazonaws.com/token
        - Form body: grant_type=refresh_token&client_id=...&client_secret=...&refresh_token=...
        - Requires clientId and clientSecret from credentials file
    """
    KIRO_DESKTOP = "kiro_desktop"
    AWS_SSO_OIDC = "aws_sso_oidc"


class KiroAuthManager:
    """
    Manages the token lifecycle for accessing Kiro API.
    
    Supports:
    - Loading credentials from .env or JSON file
    - Automatic token refresh on expiration
    - Expiration time validation (expiresAt)
    - Saving updated tokens to file
    - Both Kiro Desktop Auth and AWS SSO OIDC (kiro-cli) authentication
    
    Attributes:
        profile_arn: AWS CodeWhisperer profile ARN
        region: AWS region
        api_host: API host for the current region
        q_host: Q API host for the current region
        fingerprint: Unique machine fingerprint
        auth_type: Type of authentication (KIRO_DESKTOP or AWS_SSO_OIDC)
    
    Example:
        >>> # Kiro Desktop Auth (default)
        >>> auth_manager = KiroAuthManager(
        ...     refresh_token="your_refresh_token",
        ...     region="us-east-1"
        ... )
        >>> token = await auth_manager.get_access_token()
        
        >>> # AWS SSO OIDC (kiro-cli) - auto-detected from credentials file
        >>> auth_manager = KiroAuthManager(
        ...     creds_file="~/.aws/sso/cache/your-cache.json"
        ... )
        >>> token = await auth_manager.get_access_token()
    """
    
    def __init__(
        self,
        refresh_token: Optional[str] = None,
        profile_arn: Optional[str] = None,
        region: str = "us-east-1",
        creds_file: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        sqlite_db: Optional[str] = None,
    ):
        """
        Initializes the authentication manager.
        
        Args:
            refresh_token: Refresh token for obtaining access token
            profile_arn: AWS CodeWhisperer profile ARN
            region: AWS region (default: us-east-1)
            creds_file: Path to JSON file with credentials (optional)
            client_id: OAuth client ID (for AWS SSO OIDC, optional)
            client_secret: OAuth client secret (for AWS SSO OIDC, optional)
            sqlite_db: Path to kiro-cli SQLite database (optional)
                       Default location: ~/.local/share/kiro-cli/data.sqlite3
        """
        self._refresh_token = refresh_token
        self._profile_arn = profile_arn
        self._region = region
        self._creds_file = creds_file
        self._sqlite_db = sqlite_db
        
        # AWS SSO OIDC specific fields
        self._client_id: Optional[str] = client_id
        self._client_secret: Optional[str] = client_secret
        self._scopes: Optional[list] = None  # OAuth scopes for AWS SSO OIDC
        self._sso_region: Optional[str] = None  # SSO region for OIDC token refresh (may differ from API region)
        
        # Enterprise Kiro IDE specific fields
        self._client_id_hash: Optional[str] = None  # clientIdHash from Enterprise Kiro IDE
        
        # Track which SQLite key we loaded credentials from (for saving back to correct location)
        self._sqlite_token_key: Optional[str] = None
        
        self._access_token: Optional[str] = None
        self._expires_at: Optional[datetime] = None
        self._lock = asyncio.Lock()
        
        # Auth type will be determined after loading credentials
        self._auth_type: AuthType = AuthType.KIRO_DESKTOP
        
        # Dynamic URLs based on region
        self._refresh_url = get_kiro_refresh_url(region)
        self._api_host = get_kiro_api_host(region)
        self._q_host = get_kiro_q_host(region)
        
        # Log initialized endpoints for diagnostics (helps with DNS issues like #58)
        logger.info(f"Auth manager initialized: region={region}, api_host={self._api_host}, q_host={self._q_host}")
        
        # Fingerprint for User-Agent
        self._fingerprint = get_machine_fingerprint()
        
        # Load credentials from SQLite if specified (takes priority over JSON)
        if sqlite_db:
            self._load_credentials_from_sqlite(sqlite_db)
        # Load credentials from JSON file if specified
        elif creds_file:
            self._load_credentials_from_file(creds_file)
        
        # Determine auth type based on available credentials
        self._detect_auth_type()
    
    def _detect_auth_type(self) -> None:
        """
        Detects authentication type based on available credentials.
        
        AWS SSO OIDC credentials contain clientId and clientSecret.
        Kiro Desktop credentials do not contain these fields.
        """
        if self._client_id and self._client_secret:
            self._auth_type = AuthType.AWS_SSO_OIDC
            logger.info("Detected auth type: AWS SSO OIDC (kiro-cli)")
        else:
            self._auth_type = AuthType.KIRO_DESKTOP
            logger.info("Detected auth type: Kiro Desktop")
    
    def _load_credentials_from_sqlite(self, db_path: str) -> None:
        """
        Loads credentials from kiro-cli SQLite database.
        
        The database contains an auth_kv table with key-value pairs.
        Supports multiple authentication types:
        
        Token keys (searched in priority order):
        - 'kirocli:social:token': Social login (Google, GitHub, etc.)
        - 'kirocli:odic:token': AWS SSO OIDC (kiro-cli corporate)
        - 'codewhisperer:odic:token': Legacy AWS SSO OIDC
        
        Device registration keys (for AWS SSO OIDC only):
        - 'kirocli:odic:device-registration': Client ID and secret
        - 'codewhisperer:odic:device-registration': Legacy format
        
        The method remembers which key was used for loading, so credentials
        can be saved back to the correct location after refresh.
        
        Args:
            db_path: Path to SQLite database file
        """
        try:
            path = Path(db_path).expanduser()
            if not path.exists():
                logger.warning(f"SQLite database not found: {db_path}")
                return
            
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            
            # Try all possible token keys in priority order
            token_row = None
            for key in SQLITE_TOKEN_KEYS:
                cursor.execute("SELECT value FROM auth_kv WHERE key = ?", (key,))
                token_row = cursor.fetchone()
                if token_row:
                    self._sqlite_token_key = key  # Remember which key we loaded from
                    logger.debug(f"Loaded credentials from SQLite key: {key}")
                    break
            
            if token_row:
                token_data = json.loads(token_row[0])
                if token_data:
                    # Load token fields (using snake_case as in Rust struct)
                    if 'access_token' in token_data:
                        self._access_token = token_data['access_token']
                    if 'refresh_token' in token_data:
                        self._refresh_token = token_data['refresh_token']
                    if 'profile_arn' in token_data:
                        self._profile_arn = token_data['profile_arn']
                    if 'region' in token_data:
                        # Store SSO region for OIDC token refresh only
                        # IMPORTANT: CodeWhisperer API is only available in us-east-1,
                        # so we don't update _api_host and _q_host here.
                        # The SSO region (e.g., ap-southeast-1) is only used for OIDC token refresh.
                        self._sso_region = token_data['region']
                        logger.debug(f"SSO region from SQLite: {self._sso_region} (API stays at {self._region})")
                    
                    # Load scopes if available
                    if 'scopes' in token_data:
                        self._scopes = token_data['scopes']
                    
                    # Parse expires_at (RFC3339 format)
                    if 'expires_at' in token_data:
                        try:
                            expires_str = token_data['expires_at']
                            # Handle various ISO 8601 formats
                            if expires_str.endswith('Z'):
                                self._expires_at = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
                            else:
                                self._expires_at = datetime.fromisoformat(expires_str)
                        except Exception as e:
                            logger.warning(f"Failed to parse expires_at from SQLite: {e}")
            
            # Load device registration (client_id, client_secret) - try all possible keys
            registration_row = None
            for key in SQLITE_REGISTRATION_KEYS:
                cursor.execute("SELECT value FROM auth_kv WHERE key = ?", (key,))
                registration_row = cursor.fetchone()
                if registration_row:
                    logger.debug(f"Loaded device registration from SQLite key: {key}")
                    break
            
            if registration_row:
                registration_data = json.loads(registration_row[0])
                if registration_data:
                    if 'client_id' in registration_data:
                        self._client_id = registration_data['client_id']
                    if 'client_secret' in registration_data:
                        self._client_secret = registration_data['client_secret']
                    # SSO region from registration (fallback if not in token data)
                    if 'region' in registration_data and not self._sso_region:
                        self._sso_region = registration_data['region']
                        logger.debug(f"SSO region from device-registration: {self._sso_region}")
            
            conn.close()
            logger.info(f"Credentials loaded from SQLite database: {db_path}")
            
            # Initial cleanup if enabled
            if KIRO_CLI_DB_AUTO_CLEANUP:
                self.cleanup_sqlite_db(db_path)
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error loading credentials: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in SQLite data: {e}")
        except Exception as e:
            logger.error(f"Error loading credentials from SQLite: {e}")
    
    def _load_credentials_from_file(self, file_path: str) -> None:
        """
        Loads credentials from a JSON file.
        
        Supported JSON fields (Kiro Desktop):
        - refreshToken: Refresh token
        - accessToken: Access token (if already available)
        - profileArn: Profile ARN
        - region: AWS region
        - expiresAt: Token expiration time (ISO 8601)
        
        Additional fields for AWS SSO OIDC (kiro-cli):
        - clientId: OAuth client ID
        - clientSecret: OAuth client secret
        
        For Enterprise Kiro IDE:
        - clientIdHash: Hash of client ID (Enterprise Kiro IDE)
        - When clientIdHash is present, automatically loads clientId and clientSecret
          from ~/.aws/sso/cache/{clientIdHash}.json (device registration file)
        
        Args:
            file_path: Path to JSON file
        """
        try:
            path = Path(file_path).expanduser()
            if not path.exists():
                logger.warning(f"Credentials file not found: {file_path}")
                return
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load common data from file
            if 'refreshToken' in data:
                self._refresh_token = data['refreshToken']
            if 'accessToken' in data:
                self._access_token = data['accessToken']
            if 'profileArn' in data:
                self._profile_arn = data['profileArn']
            if 'region' in data:
                self._region = data['region']
                # Update URLs for new region
                self._refresh_url = get_kiro_refresh_url(self._region)
                self._api_host = get_kiro_api_host(self._region)
                self._q_host = get_kiro_q_host(self._region)
                logger.info(f"Region updated from credentials file: region={self._region}, api_host={self._api_host}, q_host={self._q_host}")
            
            # Load clientIdHash and device registration for Enterprise Kiro IDE
            if 'clientIdHash' in data:
                self._client_id_hash = data['clientIdHash']
                self._load_enterprise_device_registration(self._client_id_hash)
            
            # Load AWS SSO OIDC specific fields (if directly in credentials file)
            if 'clientId' in data:
                self._client_id = data['clientId']
            if 'clientSecret' in data:
                self._client_secret = data['clientSecret']
            
            # Parse expiresAt
            if 'expiresAt' in data:
                try:
                    expires_str = data['expiresAt']
                    # Support for different date formats
                    if expires_str.endswith('Z'):
                        self._expires_at = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
                    else:
                        self._expires_at = datetime.fromisoformat(expires_str)
                except Exception as e:
                    logger.warning(f"Failed to parse expiresAt: {e}")
            
            logger.info(f"Credentials loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading credentials from file: {e}")
    
    def _load_enterprise_device_registration(self, client_id_hash: str) -> None:
        """
        Loads clientId and clientSecret from Enterprise Kiro IDE device registration file.
        
        Enterprise Kiro IDE uses AWS SSO OIDC authentication. Device registration is stored at:
        ~/.aws/sso/cache/{clientIdHash}.json
        
        Args:
            client_id_hash: Client ID hash used to locate the device registration file
        """
        try:
            device_reg_path = Path.home() / ".aws" / "sso" / "cache" / f"{client_id_hash}.json"
            
            if not device_reg_path.exists():
                logger.warning(f"Enterprise device registration file not found: {device_reg_path}")
                return
            
            with open(device_reg_path, 'r', encoding='utf-8') as f:
                device_data = json.load(f)
            
            if 'clientId' in device_data:
                self._client_id = device_data['clientId']
            
            if 'clientSecret' in device_data:
                self._client_secret = device_data['clientSecret']
            
            logger.info(f"Enterprise device registration loaded from {device_reg_path}")
            
        except Exception as e:
            logger.error(f"Error loading enterprise device registration: {e}")
    
    def _save_credentials_to_file(self) -> None:
        """
        Saves updated credentials to a JSON file.
        
        Updates the existing file while preserving other fields.
        """
        if not self._creds_file:
            return
        
        try:
            path = Path(self._creds_file).expanduser()
            
            # Read existing data
            existing_data = {}
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Update data
            existing_data['accessToken'] = self._access_token
            existing_data['refreshToken'] = self._refresh_token
            if self._expires_at:
                existing_data['expiresAt'] = self._expires_at.isoformat()
            if self._profile_arn:
                existing_data['profileArn'] = self._profile_arn
            
            # Save
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Credentials saved to {self._creds_file}")
            
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
    
    def _save_credentials_to_sqlite(self) -> None:
        """
        Saves updated credentials back to SQLite database.
        
        This ensures that tokens refreshed by the gateway are persisted
        and available after gateway restart or for other processes reading
        the same SQLite database.
        
        Strategy:
        1. If we know which key we loaded from (_sqlite_token_key), save to that key
        2. If that fails or key is unknown, try all supported keys as fallback
        
        This approach ensures credentials are saved to the correct location
        regardless of authentication type (social login, AWS SSO OIDC, legacy).
        
        Updates the auth_kv table with fresh access_token, refresh_token,
        and expires_at values after successful token refresh.
        """
        if not self._sqlite_db:
            return
        
        try:
            path = Path(self._sqlite_db).expanduser()
            if not path.exists():
                logger.warning(f"SQLite database not found for writing: {self._sqlite_db}")
                return
            
            # Use timeout to avoid blocking if database is locked
            conn = sqlite3.connect(str(path), timeout=5.0)
            cursor = conn.cursor()
            
            # Prepare token data matching the structure from _load_credentials_from_sqlite
            token_data = {
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "expires_at": self._expires_at.isoformat() if self._expires_at else None,
                "region": self._sso_region or self._region,
            }
            if self._scopes:
                token_data["scopes"] = self._scopes
            
            token_json = json.dumps(token_data)
            
            # Save back to the same key we loaded from (if known)
            if self._sqlite_token_key:
                cursor.execute(
                    "UPDATE auth_kv SET value = ? WHERE key = ?",
                    (token_json, self._sqlite_token_key)
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    logger.debug(f"Credentials saved to SQLite key: {self._sqlite_token_key}")
                    return
                else:
                    logger.warning(f"Failed to update SQLite key: {self._sqlite_token_key}, trying fallback")
            
            # Fallback: try all keys (for edge cases where source key is unknown)
            for key in SQLITE_TOKEN_KEYS:
                cursor.execute(
                    "UPDATE auth_kv SET value = ? WHERE key = ?",
                    (token_json, key)
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    logger.debug(f"Credentials saved to SQLite key: {key} (fallback)")
                    return
            
            # If we get here, no keys were updated
            conn.close()
            logger.warning(f"Failed to save credentials to SQLite: no matching keys found")
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving credentials: {e}")
        except Exception as e:
            logger.error(f"Error saving credentials to SQLite: {e}")
        
        # Auto cleanup if enabled
        if KIRO_CLI_DB_AUTO_CLEANUP:
            self.cleanup_sqlite_db(self._sqlite_db)

    @staticmethod
    def cleanup_sqlite_db(db_path: str) -> bool:
        """
        Cleanup SQLite database by removing all tables except auth_kv.
        
        This removes all data unrelated to authentication, keeping only
        the auth_kv table with credentials.
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            path = Path(db_path).expanduser()
            if not path.exists():
                logger.warning(f"SQLite database not found: {db_path}")
                return False
            
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
            tables = cursor.fetchall()
            
            dropped = []
            for (table_name,) in tables:
                if table_name != 'auth_kv':
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    dropped.append(table_name)
            
            conn.commit()
            conn.close()
            
            if dropped:
                logger.info(f"Cleaned up SQLite database: dropped tables {dropped}")
            
            return True
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error during cleanup: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cleaning up SQLite database: {e}")
            return False
    
    def is_token_expiring_soon(self) -> bool:
        """
        Checks if the token is expiring soon.
        
        Returns:
            True if the token expires within TOKEN_REFRESH_THRESHOLD seconds
            or if expiration time information is not available
        """
        if not self._expires_at:
            return True  # If no expiration info available, assume refresh is needed
        
        now = datetime.now(timezone.utc)
        threshold = now.timestamp() + TOKEN_REFRESH_THRESHOLD
        
        return self._expires_at.timestamp() <= threshold
    
    def is_token_expired(self) -> bool:
        """
        Checks if the token is actually expired (not just expiring soon).
        
        This is used for graceful degradation when refresh fails but
        the access token might still be valid for a short time.
        
        Returns:
            True if the token has already expired or if expiration time
            information is not available
        """
        if not self._expires_at:
            return True  # If no expiration info available, assume expired
        
        now = datetime.now(timezone.utc)
        return now >= self._expires_at
    
    async def _refresh_token_request(self) -> None:
        """
        Performs a token refresh request.
        
        Routes to appropriate refresh method based on auth type:
        - KIRO_DESKTOP: Uses Kiro Desktop Auth endpoint
        - AWS_SSO_OIDC: Uses AWS SSO OIDC endpoint
        
        Raises:
            ValueError: If refresh token is not set or response doesn't contain accessToken
            httpx.HTTPError: On HTTP request error
        """
        if self._auth_type == AuthType.AWS_SSO_OIDC:
            await self._refresh_token_aws_sso_oidc()
        else:
            await self._refresh_token_kiro_desktop()
    
    async def _refresh_token_kiro_desktop(self) -> None:
        """
        Refreshes token using Kiro Desktop Auth endpoint.
        
        Endpoint: https://prod.{region}.auth.desktop.kiro.dev/refreshToken
        Method: POST
        Content-Type: application/json
        Body: {"refreshToken": "..."}
        
        Raises:
            ValueError: If refresh token is not set or response doesn't contain accessToken
            httpx.HTTPError: On HTTP request error
        """
        if not self._refresh_token:
            raise ValueError("Refresh token is not set")
        
        logger.info("Refreshing Kiro token via Kiro Desktop Auth...")
        
        payload = {'refreshToken': self._refresh_token}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"KiroIDE-0.7.45-{self._fingerprint}",
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self._refresh_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        
        new_access_token = data.get("accessToken")
        new_refresh_token = data.get("refreshToken")
        expires_in = data.get("expiresIn", 3600)
        new_profile_arn = data.get("profileArn")
        
        if not new_access_token:
            raise ValueError(f"Response does not contain accessToken: {data}")
        
        # Update data
        self._access_token = new_access_token
        if new_refresh_token:
            self._refresh_token = new_refresh_token
        if new_profile_arn:
            self._profile_arn = new_profile_arn
        
        # Calculate expiration time with buffer (minus 60 seconds)
        self._expires_at = datetime.now(timezone.utc).replace(microsecond=0)
        self._expires_at = datetime.fromtimestamp(
            self._expires_at.timestamp() + expires_in - 60,
            tz=timezone.utc
        )
        
        logger.info(f"Token refreshed via Kiro Desktop Auth, expires: {self._expires_at.isoformat()}")
        
        # Save to file or SQLite depending on configuration
        if self._sqlite_db:
            self._save_credentials_to_sqlite()
        else:
            self._save_credentials_to_file()
    
    async def _refresh_token_aws_sso_oidc(self) -> None:
        """
        Refreshes token using AWS SSO OIDC endpoint.
        
        Used by kiro-cli which authenticates via AWS IAM Identity Center.
        
        Strategy: Try with current in-memory token first. If it fails with 400
        (invalid_request - token was invalidated by kiro-cli re-login), reload
        credentials from SQLite and retry once.
        
        This approach handles both scenarios:
        1. Container successfully refreshed token (uses in-memory token)
        2. kiro-cli re-login invalidated token (reloads from SQLite on failure)
        
        Endpoint: https://oidc.{region}.amazonaws.com/token
        Method: POST
        Content-Type: application/x-www-form-urlencoded
        Body: grant_type=refresh_token&client_id=...&client_secret=...&refresh_token=...
        
        Raises:
            ValueError: If required credentials are not set
            httpx.HTTPError: On HTTP request error
        """
        try:
            await self._do_aws_sso_oidc_refresh()
        except httpx.HTTPStatusError as e:
            # 400 = invalid_request, likely stale token after kiro-cli re-login
            if e.response.status_code == 400 and self._sqlite_db:
                logger.warning("Token refresh failed with 400, reloading credentials from SQLite and retrying...")
                self._load_credentials_from_sqlite(self._sqlite_db)
                await self._do_aws_sso_oidc_refresh()
            else:
                raise
    
    async def _do_aws_sso_oidc_refresh(self) -> None:
        """
        Performs the actual AWS SSO OIDC token refresh.
        
        This is the internal implementation called by _refresh_token_aws_sso_oidc().
        It performs a single refresh attempt with current in-memory credentials.
        
        Uses AWS SSO OIDC CreateToken API format:
        - Content-Type: application/json (not form-urlencoded)
        - Parameter names: camelCase (clientId, not client_id)
        - Payload: JSON object
        
        Raises:
            ValueError: If required credentials are not set
            httpx.HTTPStatusError: On HTTP error (including 400 for invalid token)
        """
        if not self._refresh_token:
            raise ValueError("Refresh token is not set")
        if not self._client_id:
            raise ValueError("Client ID is not set (required for AWS SSO OIDC)")
        if not self._client_secret:
            raise ValueError("Client secret is not set (required for AWS SSO OIDC)")
        
        logger.info("Refreshing Kiro token via AWS SSO OIDC...")
        
        # AWS SSO OIDC CreateToken API uses JSON with camelCase parameters
        # Use SSO region for OIDC endpoint (may differ from API region)
        sso_region = self._sso_region or self._region
        url = get_aws_sso_oidc_url(sso_region)
        
        # IMPORTANT: AWS SSO OIDC CreateToken API requires:
        # 1. JSON payload (not form-urlencoded)
        # 2. camelCase parameter names (clientId, not client_id)
        payload = {
            "grantType": "refresh_token",
            "clientId": self._client_id,
            "clientSecret": self._client_secret,
            "refreshToken": self._refresh_token,
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Log request details (without secrets) for debugging
        logger.debug(f"AWS SSO OIDC refresh request: url={url}, sso_region={sso_region}, "
                     f"api_region={self._region}, client_id={self._client_id[:8]}...")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            # Log response details for debugging (especially on errors)
            if response.status_code != 200:
                error_body = response.text
                logger.error(f"AWS SSO OIDC refresh failed: status={response.status_code}, "
                             f"body={error_body}")
                # Try to parse AWS error for more details
                try:
                    error_json = response.json()
                    error_code = error_json.get("error", "unknown")
                    error_desc = error_json.get("error_description", "no description")
                    logger.error(f"AWS SSO OIDC error details: error={error_code}, "
                                 f"description={error_desc}")
                except Exception:
                    pass  # Body wasn't JSON, already logged as text
                response.raise_for_status()
            
            result = response.json()
        
        # AWS SSO OIDC CreateToken API returns camelCase fields
        new_access_token = result.get("accessToken")
        new_refresh_token = result.get("refreshToken")
        expires_in = result.get("expiresIn", 3600)
        
        if not new_access_token:
            raise ValueError(f"AWS SSO OIDC response does not contain accessToken: {result}")
        
        # Update data
        self._access_token = new_access_token
        if new_refresh_token:
            self._refresh_token = new_refresh_token
        
        # Calculate expiration time with buffer (minus 60 seconds)
        self._expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
        
        logger.info(f"Token refreshed via AWS SSO OIDC, expires: {self._expires_at.isoformat()}")
        
        # Save to file or SQLite depending on configuration
        if self._sqlite_db:
            self._save_credentials_to_sqlite()
        else:
            self._save_credentials_to_file()
    
    async def get_access_token(self) -> str:
        """
        Returns a valid access_token, refreshing it if necessary.
        
        Thread-safe method using asyncio.Lock.
        Automatically refreshes the token if it has expired or is about to expire.
        
        For SQLite mode (kiro-cli): implements graceful degradation when refresh fails.
        If kiro-cli has been running and refreshing tokens in memory (without persisting
        to SQLite), the refresh_token in SQLite becomes stale. In this case, we fall back
        to using the access_token directly until it actually expires.
        
        Returns:
            Valid access token
        
        Raises:
            ValueError: If unable to obtain access token
        """
        async with self._lock:
            # Token is valid and not expiring soon - just return it
            if self._access_token and not self.is_token_expiring_soon():
                return self._access_token
            
            # SQLite mode: reload credentials first, kiro-cli might have updated them
            if self._sqlite_db and self.is_token_expiring_soon():
                logger.debug("SQLite mode: reloading credentials before refresh attempt")
                self._load_credentials_from_sqlite(self._sqlite_db)
                # Check if reloaded token is now valid
                if self._access_token and not self.is_token_expiring_soon():
                    logger.debug("SQLite reload provided fresh token, no refresh needed")
                    return self._access_token
            
            # Try to refresh the token
            try:
                await self._refresh_token_request()
            except httpx.HTTPStatusError as e:
                # Graceful degradation for SQLite mode when refresh fails twice
                # This happens when kiro-cli refreshed tokens in memory without persisting
                if e.response.status_code == 400 and self._sqlite_db:
                    logger.warning(
                        "Token refresh failed with 400 after SQLite reload. "
                        "This may happen if kiro-cli refreshed tokens in memory without persisting."
                    )
                    # Check if access_token is still usable
                    if self._access_token and not self.is_token_expired():
                        logger.warning(
                            "Using existing access_token until it expires. "
                            "Run 'kiro-cli login' when convenient to refresh credentials."
                        )
                        return self._access_token
                    else:
                        raise ValueError(
                            "Token expired and refresh failed. "
                            "Please run 'kiro-cli login' to refresh your credentials."
                        )
                # Non-SQLite mode or non-400 error - propagate the exception
                raise
            except Exception:
                # For any other exception, propagate it
                raise
            
            if not self._access_token:
                raise ValueError("Failed to obtain access token")
            
            return self._access_token
    
    async def force_refresh(self) -> str:
        """
        Forces a token refresh.
        
        Used when receiving a 403 error from the API.
        
        Returns:
            New access token
        """
        async with self._lock:
            await self._refresh_token_request()
            return self._access_token
    
    @property
    def profile_arn(self) -> Optional[str]:
        """AWS CodeWhisperer profile ARN."""
        return self._profile_arn
    
    @property
    def region(self) -> str:
        """AWS region."""
        return self._region
    
    @property
    def api_host(self) -> str:
        """API host for the current region."""
        return self._api_host
    
    @property
    def q_host(self) -> str:
        """Q API host for the current region."""
        return self._q_host
    
    @property
    def fingerprint(self) -> str:
        """Unique machine fingerprint."""
        return self._fingerprint
    
    @property
    def auth_type(self) -> AuthType:
        """Authentication type (KIRO_DESKTOP or AWS_SSO_OIDC)."""
        return self._auth_type

class AccountLoadBalancer:
    """
    Load balancer for multiple Kiro accounts.
    
    Manages multiple KiroAuthManager instances and distributes requests
    across them using configurable strategies.
    
    Supported strategies:
    - round_robin: Cycle through accounts sequentially
    - random: Randomly select an account
    - failover: Use first account, failover to next on error
    
    Supports mixing different account types:
    - refresh_token: Direct refresh token configuration
    - creds_file: JSON credentials file (Kiro Desktop or AWS SSO)
    - sqlite_db: kiro-cli SQLite database
    
    Attributes:
        accounts: List of KiroAuthManager instances
        strategy: Load balancing strategy
        round_robin_index: Current index for round_robin strategy
    """
    
    def __init__(
        self,
        accounts_json: str,
        strategy: str = "round_robin",
        default_region: str = "us-east-1"
    ):
        """
        Initialize load balancer with account configurations.
        
        Args:
            accounts_json: JSON string containing account configurations
            strategy: Load balancing strategy (round_robin, random, failover)
            default_region: Default AWS region if not specified in account config
        """
        self.accounts: list[KiroAuthManager] = []
        self.strategy = strategy
        self.round_robin_index = 0
        self._lock = asyncio.Lock()
        
        self._parse_accounts(accounts_json, default_region)
    
    def _parse_accounts(self, accounts_json: str, default_region: str) -> None:
        """Parse JSON configuration and create auth managers."""
        import json
        
        try:
            accounts_data = json.loads(accounts_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse KIRO_ACCOUNTS_JSON: {e}")
            return
        
        if not accounts_data:
            logger.warning("KIRO_ACCOUNTS_JSON is empty")
            return
        
        for i, account_config in enumerate(accounts_data):
            try:
                account_type = account_config.get("type", "refresh_token")
                region = account_config.get("region", default_region)
                
                if account_type == "refresh_token":
                    refresh_token = account_config.get("refresh_token", "")
                    profile_arn = account_config.get("profile_arn", "")
                    auth_manager = KiroAuthManager(
                        refresh_token=refresh_token,
                        profile_arn=profile_arn,
                        region=region
                    )
                elif account_type == "creds_file":
                    creds_file = account_config.get("creds_file", "")
                    auth_manager = KiroAuthManager(
                        creds_file=creds_file,
                        region=region
                    )
                elif account_type == "sqlite_db":
                    sqlite_db = account_config.get("sqlite_db", "")
                    auth_manager = KiroAuthManager(
                        sqlite_db=sqlite_db,
                        region=region
                    )
                else:
                    logger.warning(f"Unknown account type: {account_type}")
                    continue
                
                self.accounts.append(auth_manager)
                logger.info(f"Account {i+1} loaded: type={account_type}, region={region}")
                
            except Exception as e:
                logger.error(f"Failed to load account {i+1}: {e}")
        
        logger.info(f"Load balancer initialized with {len(self.accounts)} accounts, strategy={self.strategy}")
    
    def get_account(self) -> KiroAuthManager:
        """
        Get an auth manager based on load balancing strategy.
        
        Returns:
            KiroAuthManager instance
            
        Raises:
            ValueError: If no accounts are configured
        """
        if not self.accounts:
            raise ValueError("No accounts configured in load balancer")
        
        if self.strategy == "random":
            import random
            return random.choice(self.accounts)
        
        elif self.strategy == "failover":
            return self.accounts[0]
        
        else:  # round_robin (default)
            async def get_next():
                async with self._lock:
                    account = self.accounts[self.round_robin_index]
                    self.round_robin_index = (self.round_robin_index + 1) % len(self.accounts)
                    return account
            # For sync context, use simple round-robin without lock
            account = self.accounts[self.round_robin_index]
            self.round_robin_index = (self.round_robin_index + 1) % len(self.accounts)
            return account
    
    async def get_account_async(self) -> KiroAuthManager:
        """
        Get an auth manager asynchronously (thread-safe for round-robin).
        
        Returns:
            KiroAuthManager instance
        """
        if not self.accounts:
            raise ValueError("No accounts configured in load balancer")
        
        if self.strategy == "random":
            import random
            return random.choice(self.accounts)
        
        elif self.strategy == "failover":
            return self.accounts[0]
        
        else:  # round_robin (default)
            async with self._lock:
                account = self.accounts[self.round_robin_index]
                self.round_robin_index = (self.round_robin_index + 1) % len(self.accounts)
                return account
    
    def failover_to_next(self, failed_account: KiroAuthManager) -> KiroAuthManager:
        """
        Move failed account to end and return next account (for failover strategy).
        
        Args:
            failed_account: The account that failed
            
        Returns:
            Next KiroAuthManager instance
        """
        if len(self.accounts) <= 1:
            return failed_account
        
        try:
            self.accounts.remove(failed_account)
            self.accounts.append(failed_account)
            logger.info("Failed account moved to end of queue")
        except ValueError:
            pass
        
        return self.accounts[0]
    
    @property
    def account_count(self) -> int:
        """Return number of configured accounts."""
        return len(self.accounts)
