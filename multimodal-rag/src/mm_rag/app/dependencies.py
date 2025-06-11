from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


auth_pat_dependency = HTTPBearer(description="Enter your personal authorization token:")
