# OpenTrace Documentation

This directory contains the MkDocs Material documentation for OpenTrace.

## 🌿 Branch Workflow

### Development (Local Only)
- **Branch**: `docs-dev`
- **URL**: Local only - `http://127.0.0.1:8000/`
- **Purpose**: Intermediate edits, testing, review
- **Features**: Local development with live-reload

### Production (Live Site)
- **Branch**: `docs-prod` 
- **URL**: https://agentopt.github.io/OpenTrace
- **Purpose**: Public documentation that users see
- **Features**: Deployed to GitHub Pages, production analytics

## 🚀 Quick Start with Makefile

The easiest way to work with OpenTrace documentation is using the included Makefile commands from the project root:

### ⚡ Essential Commands
```bash
# Start local development server
make serve

# Publish documentation (full deployment pipeline)
make publish

# See all available commands
make help
```

### 📋 Complete Command Reference
```bash
# 🚀 Development Commands
make serve          # Start local server (http://127.0.0.1:8000)
make build          # Build documentation locally  
make clean          # Clean build artifacts

# 📦 Publishing Commands
make publish        # One-command publish to GitHub Pages
make status         # Show current branch and git status

# 🌿 Branch Management
make switch-dev     # Switch to docs-dev branch (staging)
make switch-prod    # Switch to docs-prod branch (production)  

# 🔧 Setup Commands
make install        # Install documentation dependencies
make help           # Display help with all commands
```

### 🎯 Recommended Workflow
```bash
# 1. Start on development branch
make switch-dev

# 2. Start local server for live preview
make serve          # Visit http://127.0.0.1:8000

# 3. Make your changes...
# (edit files in docs-mkdocs/docs/)

# 4. When ready, publish everything
make publish        # Handles entire deployment pipeline automatically
```

## 🛠️ Manual Development (Alternative)

If you prefer working directly with MkDocs commands:

### Setup
```bash
cd docs-mkdocs
make install        # or: pip install mkdocs-material mkdocs-jupyter
```

### Serve Locally
```bash
cd docs-mkdocs
mkdocs serve        # Visit http://127.0.0.1:8000/
```

### Manual Deployment Process

### 1. Development Work
```bash
# Work on docs-dev branch
git checkout docs-dev
# Make your changes...
git add .
git commit -m "Update documentation"
git push origin docs-dev
```

### 2. Preview Changes Locally
- Run `mkdocs serve` to preview at http://127.0.0.1:8000/
- Review changes with live-reload
- Test all functionality locally

### 3. Publish to Production
```bash
# When ready to publish
git checkout docs-prod
git merge docs-dev
git push origin docs-prod
```

## 📁 Structure

```
docs-mkdocs/
├── mkdocs.yml          # Main configuration
├── docs/               # Documentation content
│   ├── index.md        # Homepage
│   ├── quickstart/     # Learning materials  
│   ├── tutorials/      # In-depth guides
│   ├── examples/       # Code examples
│   ├── stylesheets/    # Custom CSS
│   └── images/         # Assets
└── site/              # Built documentation (auto-generated)
```

## 📖 API Documentation

The API documentation is automatically generated from Python docstrings using:
- **mkdocstrings** - Renders docstrings into documentation pages  
- **gen_ref_pages.py** - Scans the `opto` package and creates API structure
- **Auto-generation** - Runs automatically on `make serve`, `make build`, and `make publish`

No manual steps needed - just update your docstrings in the source code and the API docs will reflect the changes! The generated API documentation includes:
- Complete module reference for `opto.trace`, `opto.optimizers`, `opto.trainer`, etc.
- Function signatures with type hints
- Class documentation with methods and attributes
- Cross-references between related components

## 🎨 Customization

- **Colors**: Defined in `docs/stylesheets/extra.css`
- **Theme**: Material theme with custom OpenTrace branding
- **Fonts**: Circular Std with fallbacks
- **Analytics**: Google Analytics (G-C3WH29YM90)

## 📊 Analytics

- **Production**: Full Google Analytics tracking
- **Staging**: Same analytics (tagged as staging environment)

## 🔧 GitHub Actions

The workflow automatically:
- Builds documentation on push to `docs-dev` or `docs-prod`
- Deploys to appropriate URLs
- Adds staging banner for dev environment
- Manages concurrent deployments per branch