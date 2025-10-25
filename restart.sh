pid=$(ps aux | grep main_api.py  | grep -v grep | awk '{print $2}')
echo "pid:$pid"

if [ -n "$pid" ];then
  echo "kill pid:$pid"
  kill -9  $pid
fi
( nohup python main_api.py  1>&2 > run_log.log  & )
