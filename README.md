gpt-flux-ouroboros
==================

Patrick Wagstrom &lt;160672+pridkett@users.noreply.github.com&gt;

August 2024

Overview
========

This project is inspired by [this Reddit post, where /u/lazyspock used ChatGPT 4o to describe pictures and then fed that to Flux for image generation](https://old.reddit.com/r/StableDiffusion/comments/1elb3mp/got_pictures_from_pexelscom_asked_chatgpt_4o_to/). Being inspired and having some free time on my hands, I decided to see if I could turn that into a node for ComfyUI, so you can use GPT-4o to generate your prompts and then do whatever else you want to do with it.

Setup
=====

This uses pipenv to manage the dependencies. You'll need to do something like this:

```bash
pipenv --python 3.11
pipenv install
```

From there you can use `pipenv shell` to get into the environment, or you can configure your IDE of choice to make use of the pipenv environment.

Usage
=====

License
=======

Copyright (c) 2024 Patrick Wagstrom

Licensed under terms of the MIT License
