#!/usr/bin/env python
# -*- coding: utf-8 -*-

import launch

if not launch.is_installed("openai"):
    launch.run_pip("install openai", "requirements iF_prompt_MKR")