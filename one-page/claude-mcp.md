# Documentation

Claude Model Context Protocol  
https://modelcontextprotocol.io/introduction  

Git  
https://github.com/modelcontextprotocol  

# Install

Chocolatey  
https://chocolatey.org/install  

Installed using Chocolatey  
https://nodejs.org/en/download/package-manager  

SQLite  
https://www.sqlite.org/download.html  

## Config

Claude enabiling MCP Servers in C:\Users\wil\AppData\Roaming\Claude

```bash
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": [
        "mcp-server-sqlite",
        "--db-path",
        "D:\\perf.db"
      ]
    },
	  "github": {
      "command": "node",
      "args": ["D:\\servers\\src\\github\\dist\\index.js"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "..."
      }
    },
    "filesystem": {
      "command": "node",
      "args": ["D:\\servers\\src\\filesystem\\dist\\index.js","d:\\"]
    },
    "slack": {
      "command": "node",
      "args": ["D:\\servers\\src\\slack\\dist\\index.js"],
      "env": {
        "SLACK_BOT_TOKEN": "...",
        "SLACK_TEAM_ID": "..."
      }
    }
  }
}
```
## Local computer

Clone Repo

```bash
git https://github.com/modelcontextprotocol/servers  
```

Build

```bash
cd servers\src
cd filesystem
npm run build
cd ..
cd github
npm run build
cd ..
cd slack
npm run build
```

Test

```bash
cd filesystem
node D:\servers\src\filesystem\dist\index.js d:\
```

## Claude

Everytime you modify the configuration, you need to quit Claude Desktop not only close the Window.  
In: File -> Exit  