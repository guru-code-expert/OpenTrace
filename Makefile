.PHONY: help serve build clean publish status switch-dev switch-prod install

help:
	@echo "OpenTrace Documentation Makefile"
	@echo ""
	@echo "🚀 Development Commands:"
	@echo "  make serve          - Start local development server (http://127.0.0.1:8000)"
	@echo "  make build          - Build documentation locally"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "📦 Publishing Commands:"
	@echo "  make publish        - Publish documentation to GitHub Pages"
	@echo "  make status         - Show current branch and git status"
	@echo ""
	@echo "🌿 Branch Management:"
	@echo "  make switch-dev     - Switch to docs-dev branch (staging)"
	@echo "  make switch-prod    - Switch to docs-prod branch (production)"
	@echo ""
	@echo "🔧 Setup Commands:"
	@echo "  make install        - Install documentation dependencies"
	@echo ""
	@echo "📝 Workflow:"
	@echo "  1. Work on docs-dev branch: make switch-dev"
	@echo "  2. Test locally: make serve"
	@echo "  3. When ready to publish: make publish"

serve:
	@echo "🚀 Starting local development server..."
	@echo "📍 URL: http://127.0.0.1:8000"
	@echo "💡 Press Ctrl+C to stop"
	@cd docs-mkdocs && mkdocs serve

build:
	@echo "🔨 Building documentation..."
	@cd docs-mkdocs && mkdocs build --clean --strict
	@echo "✅ Build complete! Output in docs-mkdocs/site/"

clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf docs-mkdocs/site/
	@echo "✅ Clean complete!"

publish:
	@echo "📦 Publishing documentation to GitHub Pages..."
	@echo ""
	@echo "Current branch: $$(git branch --show-current)"
	@echo "Current status:"
	@git status --porcelain
	@echo ""
	@if [ "$$(git branch --show-current)" != "docs-dev" ]; then \
		echo "⚠️  Warning: You're not on docs-dev branch!"; \
		echo "   Current branch: $$(git branch --show-current)"; \
		echo "   Recommended: make switch-dev first"; \
		echo ""; \
		read -p "Continue anyway? (y/N): " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "❌ Publish cancelled"; \
			exit 1; \
		fi; \
	fi
	@echo "🔍 Building documentation first..."
	@$(MAKE) build
	@echo ""
	@echo "💾 Committing changes..."
	@git add docs-mkdocs/ .github/workflows/docs.yml Makefile
	@if git diff --staged --quiet; then \
		echo "ℹ️  No changes to commit"; \
	else \
		git commit -m "Update documentation\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"; \
	fi
	@echo ""
	@echo "⬆️  Pushing docs-dev to remote..."
	@git push origin docs-dev
	@echo ""
	@echo "🔄 Switching to docs-prod and merging..."
	@git checkout docs-prod
	@git merge docs-dev --no-edit
	@echo ""
	@echo "🚀 Pushing to docs-prod (triggers GitHub Pages deployment)..."
	@git push origin docs-prod
	@echo ""
	@echo "🔄 Switching back to docs-dev..."
	@git checkout docs-dev
	@echo ""
	@echo "✅ Publish complete!"
	@echo "🌐 Your documentation will be available at: https://agentopt.github.io/OpenTrace"
	@echo "⏱️  GitHub Pages deployment usually takes 1-2 minutes"

status:
	@echo "📊 Repository Status"
	@echo "===================="
	@echo "Current branch: $$(git branch --show-current)"
	@echo "Remote URL: $$(git remote get-url origin)"
	@echo ""
	@echo "Git Status:"
	@git status --short
	@echo ""
	@echo "Recent commits:"
	@git log --oneline -5

switch-dev:
	@echo "🌿 Switching to docs-dev branch..."
	@git checkout docs-dev
	@echo "✅ Now on docs-dev branch (staging)"

switch-prod:
	@echo "🌿 Switching to docs-prod branch..."
	@git checkout docs-prod  
	@echo "✅ Now on docs-prod branch (production)"
	@echo "⚠️  Note: This branch auto-deploys to GitHub Pages on push"

install:
	@echo "🔧 Installing documentation dependencies..."
	@pip install mkdocs-material mkdocs-jupyter
	@pip install mkdocs-git-revision-date-localized-plugin
	@pip install mkdocs-git-committers-plugin-2
	@echo "✅ Dependencies installed!"
	@echo ""
	@echo "💡 Next steps:"
	@echo "   make serve    - Start development server"
	@echo "   make publish  - Publish to GitHub Pages"