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

## 🚀 Deployment Process

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

## 🛠️ Local Development

### Setup
```bash
cd docs-mkdocs
pip install mkdocs-material mkdocs-jupyter
```

### Serve Locally
```bash
mkdocs serve
# Visit http://127.0.0.1:8000/
```

### Build
```bash
mkdocs build --clean --strict
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