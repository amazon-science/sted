#!/bin/bash
# Helper script to create new results version

DATE=$(date +%Y-%m-%d)
VERSION_NAME="v${1:-$DATE}"
VERSION_DIR="results_archive/$VERSION_NAME"

echo "Creating new version: $VERSION_NAME"

# Create version directory
mkdir -p "$VERSION_DIR"/{llm_consistency,variation_analysis,dataset_analysis,experiments,archive}

# Create VERSION.md template
cat > "$VERSION_DIR/VERSION.md" << EOV
# $VERSION_NAME

## Changes
- [Describe what changed in this version]

## Improvements
- [List improvements made]

## Metrics
- [Key metrics and results]

## Notes
- [Any additional notes]

Created: $(date)
EOV

# Update symlink
ln -sf "../$VERSION_DIR" results/latest

echo "✓ Created $VERSION_DIR"
echo "✓ Updated results/latest symlink"
echo ""
echo "Next steps:"
echo "1. Edit $VERSION_DIR/VERSION.md"
echo "2. Run experiments with --output-dir $VERSION_DIR"
echo "3. Compare with previous version"
