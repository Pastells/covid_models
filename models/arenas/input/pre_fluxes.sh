#!/bin/bash
vim -c ":%s/ ;/;/g" -c ":w ++enc=utf-8" -c q $1
