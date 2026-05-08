#!/usr/bin/env python3
"""Post-render script to fix sitemap.xml canonical URLs for index pages."""
import re
from pathlib import Path

def fix_sitemap():
    sitemap_path = Path("_site/sitemap.xml")
    if not sitemap_path.exists():
        print("Sitemap not found, skipping fix.")
        return
    
    content = sitemap_path.read_text()
    
    # Fix index.html URLs to use clean directory URLs
    # Replace /index.html with /
    content = re.sub(r'<loc>(https://kareemai\.com)/index\.html</loc>', r'<loc>\1/</loc>', content)
    
    # Replace /subdir/index.html with /subdir/
    content = re.sub(r'<loc>(https://kareemai\.com/[^<]+)/index\.html</loc>', r'<loc>\1/</loc>', content)
    
    sitemap_path.write_text(content)
    print("Sitemap fixed: index.html URLs converted to clean URLs")

if __name__ == "__main__":
    fix_sitemap()
