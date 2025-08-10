# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Hexo-based static blog** using the **NexT theme** (version 8.23.0). The blog is deployed automatically to GitHub Pages via GitHub Actions and focuses on technical content covering machine learning, software engineering, and various tech topics.

## Development Commands

### Core Commands

```bash
# Install dependencies
npm install

# Start development server (default port 4000)
hexo server
# Or with custom port
hexo s -p ${PORT}

# Generate static files
npm run build
# Or
hexo generate

# Clean generated files
npm run clean
# Or
hexo clean

# Deploy (handled by GitHub Actions)
npm run deploy
```

### Content Management

```bash
# Create new post
hexo new post "post-title"

# Create new page
hexo new page "page-name"
```

### Custom Python Utilities

The blog includes custom Python utilities in `source/python/` for content management:

- `refctl.py` - Reference management CLI tool (can be symlinked to `/usr/bin/refctl`)
- `download_images.py` - Image downloading utility
- `replace_posts.py` - Post content replacement tool
- `valid_images.py` - Image validation utility
- Various other content management scripts

## Architecture & Structure

### Directory Structure

```
blog/
├── source/              # Source content
│   ├── _posts/         # Blog posts (Markdown files)
│   ├── _data/          # Theme customization files
│   ├── images/         # Post images organized by post name
│   ├── python/         # Custom Python utilities
│   ├── props/          # Static assets (favicon, profile, etc.)
│   └── scripts/        # Shell scripts
├── themes/next/        # NexT theme (submodule)
├── k8s/               # Kubernetes deployment manifests
├── _config.yml        # Main Hexo configuration
├── _config.next.yml   # NexT theme configuration
└── package.json       # Dependencies
```

### Content Organization

- **Posts**: Stored in `source/_posts/` as Markdown files
- **Images**: Organized in `source/images/[post-name]/` directories
- **Categories**: Auto-generated from post front matter
- **Tags**: Auto-generated from post front matter

### Theme Customization

The blog uses extensive NexT theme customization:

- Custom fonts: D2Coding for code, Noto Serif KR for global text
- Custom colors and branding (red accent color: #800a0a)
- Gitalk comments integration
- Google Analytics integration
- Custom CSS/JS in `source/_data/`

### Key Configuration Files

- `_config.yml`: Main Hexo configuration with site metadata, plugins, and URL aliases
- `_config.next.yml`: NexT theme configuration with UI customization, features, and third-party integrations

## Development Workflow

### Local Development

1. Clone with submodules: `git clone --recurse-submodules`
2. Install dependencies: `npm install`
3. Start dev server: `hexo s -p 4000`
4. Create content in `source/_posts/`
5. Add images to `source/images/[post-name]/`

### Content Creation

- Use front matter with title, date, categories, and tags
- Images should be placed in dedicated directories under `source/images/`
- Reference management can be handled via the `refctl.py` utility
- URL aliases are configured in `_config.yml` under the `alias` section

### Deployment

- Automatic deployment to GitHub Pages via `.github/workflows/deploy.yaml`
- Triggered on pushes to `main` branch
- Uses Node.js 20, builds with `npm run build`, and deploys to GitHub Pages

## Important Notes

- The blog uses git submodules for the NexT theme
- Custom Python utilities require appropriate dependencies (loguru, etc.)
- Images are organized by post name for better management
- The blog includes extensive URL alias mapping for SEO purposes
- GitHub Actions automatically handles Gitalk secrets injection during deployment
- The site uses Korean and English content with timezone set to Asia/Seoul

## Special Features

- **Reference Management**: Custom UUID-based reference system with LaTeX-style output
- **Custom Plugins**: Various Hexo plugins for SEO, feeds, search, and aliases
- **Analytics**: Google Analytics integration with tracking ID
- **Comments**: Gitalk integration for GitHub-based comments
- **Search**: Local search functionality with hexo-generator-searchdb
- **Deployment**: Automated CI/CD with GitHub Actions and GitHub Pages
