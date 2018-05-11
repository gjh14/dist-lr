ps aux | grep "distlr" |grep -v grep| cut -c 9-15 | xargs kill -9

