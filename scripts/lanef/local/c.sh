#!/bin/bash
colab --config "$HOME/.config/colab-cli/sessions-lanef-20261020.json" "$@" 2>&1 | grep -v "new version\|colab update\|silence this"
