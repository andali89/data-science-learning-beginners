# Data Science Learning Beginners - Project Structure & Jekyll Conversion Guide

## Project Overview

This is a Jekyll-based educational website for data science beginners, featuring a comprehensive tutorial series on Python, Anaconda, and Jupyter Notebook fundamentals.

## Project Structure

```
data-science-learning-beginners/
├── _config.yml                 # Jekyll configuration
├── _includes/                  # Reusable components
├── _layouts/                   # Page templates
├── _pages/                     # Static pages
├── _posts/                     # Blog posts (main tutorial content)
│   ├── 2024-01-01-core-concepts.md
│   ├── 2024-01-02-python-ides.md
│   ├── 2024-01-03-installation-guide.md
│   ├── 2024-01-04-anaconda-basics.md
│   ├── 2024-01-05-pip-package-management.md
│   ├── 2024-01-06-jupyter-notebook-quickstart.md
│   ├── 2024-01-07-useful-resources.md
│   └── 2024-01-08-summary.md
├── _posts_bak/                 # Example theme posts (for reference)
├── _site/                      # Generated Jekyll site
└── assets/
    └── img_ana/                # Tutorial images
```

## Jekyll Post Conversion Guidelines

### 1. Front Matter Requirements

Every Jekyll post must begin with YAML front matter:

```yaml
---
title: "Post Title"
author: Anda Li
date: YYYY-MM-DD
category: Data Science Learning
layout: post
---
```

**Key Points:**
- Use descriptive titles in Chinese (as per project requirements)
- Author must be set to "Anda Li" (as requested)
- Use sequential dates to maintain post order
- Category helps group related content
- Layout "post" applies the blog post template

### 2. File Naming Convention

Jekyll posts must follow the strict naming pattern:
```
YYYY-MM-DD-title-slug.md
```

**Examples:**
- `2024-01-01-core-concepts.md`
- `2024-01-02-python-ides.md`
- `2024-01-03-installation-guide.md`

**Important:**
- Date in filename determines post order
- Use hyphens for title slugs
- Keep English slugs for URLs even with Chinese titles

### 3. Image References

Use Jekyll's `site.baseurl` variable for images:

```markdown
![Alt text]({{ site.baseurl}}/assets/img_ana/image.png)
```

**Why this format:**
- `{{ site.baseurl}}` ensures images work in subdirectories
- Absolute paths (`/assets/...`) may break in some configurations
- Consistent with Jekyll best practices

### 4. Theme Enhancement Features

This theme supports special formatting blocks:

#### Tips
```markdown
> ##### TIP
>
> Your tip content here.
{: .block-tip }
```

#### Warnings
```markdown
> ##### WARNING
>
> Your warning content here.
{: .block-warning }
```

#### Dangers
```markdown
> ##### DANGER
>
> Your danger content here.
{: .block-danger }
```

#### Wide Tables
```markdown
<div class="table-wrapper" markdown="block">

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data     | Data     | Data     |

</div>
```

#### Footnotes
```markdown
This is text with a footnote[^1].

[^1]: This is the footnote content.
```

## Content Preservation Rules

When converting markdown to Jekyll posts:

1. **Never alter the original content or text**
2. **Only modify structure and formatting**
3. **Preserve all technical information exactly**
4. **Maintain the educational flow and sequence**

## Common Conversion Steps

### Step 1: Add Front Matter
Replace the original `# Title` with proper YAML front matter.

### Step 2: Rename Files
Use `YYYY-MM-DD-slug.md` format with sequential dates.

### Step 3: Fix Image Paths
Update all image references to use `{{ site.baseurl}}` format.

### Step 4: Apply Theme Enhancements
- Convert simple blockquotes to tip/warning/danger blocks
- Wrap wide tables in table-wrapper divs
- Add footnotes where appropriate

### Step 5: Verify Jekyll Compatibility
- Check that all liquid tags are properly formatted
- Ensure no conflicting markdown syntax
- Test image paths and links

## Asset Management

### Image Organization
- All tutorial images stored in `/assets/img_ana/`
- Use descriptive filenames (e.g., `Section6_jupyter_main.png`)
- Maintain original image organization from tutorials

### Image Naming Convention
- `1_python.png` - Core concept illustrations
- `Section4_activate.png` - Section-specific screenshots  
- `5_Anaconda_in_X.png` - Installation step screenshots

## Technical Notes

### Jekyll Processing
- Posts are processed in date order (newest first by default)
- Front matter variables can be used in layouts and includes
- Liquid templating enables dynamic content generation

### Theme Compatibility
- Based on GitBook-style Jekyll theme
- Supports syntax highlighting for code blocks
- Responsive design for mobile compatibility
- Search functionality included

### Browser Compatibility
- Tested with modern browsers
- Mobile-responsive design
- Supports code syntax highlighting
- MathJax support for mathematical expressions

## Maintenance Guidelines

### Adding New Posts
1. Follow the naming convention
2. Use sequential dates to maintain order
3. Include proper front matter
4. Use theme enhancement features appropriately

### Updating Existing Posts
1. Preserve the original filename and date
2. Update only content, not structure
3. Test image links after changes
4. Maintain content accuracy

### Asset Management
1. Store images in appropriate subdirectories
2. Use descriptive filenames
3. Optimize images for web delivery
4. Test image loading after updates

---

*Generated by Claude AI Assistant - Jekyll Conversion Documentation*
*Last Updated: September 15, 2025*