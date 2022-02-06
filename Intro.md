---
layout: page
title: Intro
permalink: /intro/
nav_order: 2
---

For Jekyll reference see [just_the_docs](https://pmarsceill.github.io/just-the-docs/)


The following pages are built in order to understand Computer Vision and Machine Learning

To deploy on heroku follow the steps in the link below (and use the gem files, rake files and proc files in this repo for reference)

The following files will need to be copied from this repo:
- config.ru
- Rakefile
- Procfile
- static.json
- config.yaml (only the differences)

And only if necessary:
- Gemfile
- Gemfile.lock
- remove _sites from .gitignore

Run bundle exec jekyll serve after making the above changes

After copying these files (or their necessary contents), install heroku cli and do:
```bash
heroku login
```

Then do heroku create as per the below link and the other steps necessary (git push heroku master)

[Deploy jekyll on heroku](https://blog.heroku.com/jekyll-on-heroku)

Finally, go to heroku page -> settings -> change the name of the app and find the url