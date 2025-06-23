import json
import os
import hashlib
from typing import Dict, Optional

class UserManager:
    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self.users = self._load_users()

    def _load_users(self) -> Dict:
        """Load users from JSON file"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            else:
                # Create empty users file
                with open(self.users_file, 'w') as f:
                    json.dump({}, f)
                return {}
        except Exception as e:
            print(f"Error loading users: {e}")
            return {}

    def _save_users(self) -> None:
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, password: str, email: str) -> bool:
        """Create a new user"""
        if username in self.users:
            return False
        
        hashed_password = self._hash_password(password)
        self.users[username] = {
            'password': hashed_password,
            'email': email
        }
        self._save_users()
        return True

    def verify_user(self, username: str, password: str) -> bool:
        """Verify user credentials"""
        if username not in self.users:
            return False
        
        hashed_password = self._hash_password(password)
        return self.users[username]['password'] == hashed_password

    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information"""
        return self.users.get(username)
