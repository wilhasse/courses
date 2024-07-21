# Project

Testing Lucia Auth  
https://lucia-auth.com

Github  
https://github.com/lucia-auth/lucia

Example for Svelte  
https://github.com/lucia-auth/examples/tree/main/sveltekit/username-and-password

Oslo  
https://oslo.js.org

# Lucia Install

```bash
npm install lucia
npm install @lucia-auth/adapter-mysql
npm install mysql2
```

## Database

```sql
CREATE TABLE user (
    id VARCHAR(255) PRIMARY KEY,
    username VARCHAR(50),
    password VARCHAR(255)
);

CREATE TABLE user_session (
    id VARCHAR(255) PRIMARY KEY,
    expires_at DATETIME NOT NULL,
    user_id VARCHAR(255) NOT NULL REFERENCES user(id)
)
```
