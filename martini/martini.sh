#!/bin/bash

TAG="aiwdp-beta.rc3"

DELAY=${DELAY:-7}

THIS_SCRIPT_DIR=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
OLIVE_DIR=$(dirname $THIS_SCRIPT_DIR)
MY_OLIVE_APP_DATA=${OLIVE_APP_DATA:-$OLIVE_DIR/oliveAppData}

OLIVE_HOST_PLUGINS=${MY_OLIVE_APP_DATA}/plugins
OLIVE_HOST_SERVER=${MY_OLIVE_APP_DATA}/server
OLIVE_HOST_WORKFLOWS=${MY_OLIVE_APP_DATA}/workflows
OLIVE_HOST_CERT_FILES=${MY_OLIVE_APP_DATA}/certs
OLIVE_HOST_DOCSITE=${OLIVE_HOST_DOCSITE:-$OLIVE_DIR/docs}
OLIVE_HOST_PORT_OLIVESERVER=${OLIVE_HOST_PORT_OLIVESERVER:-5588}
OLIVE_HOST_PORT_OLIVESECOND=${OLIVE_HOST_PORT_OLIVESECOND:-5589}
OLIVE_HOST_PORT_OLIVESTREAMING=${OLIVE_HOST_PORT_OLIVESTREAMING:-5590}
OLIVE_HOST_PORT_OLIVESTREAMINGTWO=${OLIVE_HOST_PORT_OLIVESTREAMINGTWO:-5591}
OLIVE_HOST_PORT_TEST=${OLIVE_HOST_PORT_TEST:-5004}
OLIVE_HOST_PORT_DOCSERVER=${OLIVE_HOST_PORT_DOCSERVER:-5570}
OLIVE_HOST_PORT_WEBSERVER=${OLIVE_HOST_PORT_WEBSERVER:-5580}
OLIVE_HOST_OLIVE_CONF_FILE=${MY_OLIVE_APP_DATA}/olive.conf
RAN_CMD="false"

# RUNNING_CONTAINER will be "" if no container is running.
RUNNING_CONTAINER=$(docker ps -a | grep olive-martini:$TAG | cut -f 1 -d ' ')

case ${1:-help} in

  debug)
    RAN_CMD="true"
    if [ "$RUNNING_CONTAINER" = "" ] ; then
        echo "No Martini container is running so can not run the $1 command."
        exit 2
    fi
    docker exec -it $RUNNING_CONTAINER  /bin/bash
    ;;

  help)
    RAN_CMD="true"
    cat <<-EOF
	Supported commands include: cli, help, list, log, net, start, stop, status, and version.
        martini.sh cli: Starts a shell on the container for debugging.
        martini.sh help: Prints out help information.
        martini.sh list: List the running container.
        martini.sh log: Print out log files from the container.
        martini.sh net: Shows the ports on the host that the container is listening on.
        martini.sh start: Start the container.
            Optional flags:
              --gpu: Enable Martini access to GPUs, if available. Be sure to properly configure any plugin/domains to be used with GPU devices as outlined in the documentation.
              --tls_server_only: The server will be configured with a certificate and will only respond to HTTPS requests; clients aren't required to send certificate but must use HTTPS protocol (one-way TLS)
              --tls_server_and_client: Both the server and clients will need to communicate with certificates over HTTPS (two-way TLS)
              --debug: Activate 'debug' mode for the OLIVE Server. All server logs will be maintained to aid in troubleshooting and debugging.
        martini.sh status: Shows status of the processes on the container.
        martini.sh stop: Stop the container.
        martini.sh version: Prints out version information.
	EOF
    ;;

  list)
    RAN_CMD="true"
    FULLNAME=olive-martini-$TAG
    docker ps | head -1
    docker ps | grep $FULLNAME
    ;;

  log)
    RAN_CMD="true"
    if [[ "$2" != "help" && "$2" != "--help" && "$RUNNING_CONTAINER" = "" ]] ; then
        echo "No Martini container is running so can not run the $1 command."
        exit 2
    fi
    LOGNAME="/home/olive/olive/oliveserver.log"
    case $2 in
        help|--help) cat <<-EOF
	martini.sh log <which-log-file> <-f>
	where <which-log-file> is one of:
	    message: for the Olive message log
	    olive (the default): for the Oliver server log
	    pa: for the reverse proxy (nginx) access log
	    pe: for the reverse proxy (nginx) error log
	    po: for the reverse proxy (nginx) output log
	    webui: for the web ui (http server) log
        The -f is optional but does a "tail -f" if given.
	EOF
        exit 0
	;;
        mess*|olive-mess*) LOGNAME="/home/olive/olive/messagebroker.log" ;;
        olive|olive-server) LOGNAME="/home/olive/olive/oliveserver.log" ;;
        pa) LOGNAME="/home/olive/olive/nginx_access.log" ;;
        pe) LOGNAME="/home/olive/olive/nginx_error.log" ;;
        po) LOGNAME="/home/olive/olive/nginx.log" ;;
        webui*) LOGNAME="/home/olive/olive/httpdserver.log" ;;
        martini|base) LOGNAME="/home/olive/olive/martini-process.log" ;;
    esac
    docker exec -it $RUNNING_CONTAINER tail $3 $LOGNAME -n +0
    ;;

  start)
    RAN_CMD="true"
    HAVE_DIRS=1
    if [ ! -d "$OLIVE_HOST_PLUGINS" ] ; then
        HAVE_DIRS=0
        echo "The path to plugins is not set or can not be found."
        echo "Plugins are expected here: $OLIVE_HOST_PLUGINS"
    fi
    if [ ! -d "$OLIVE_HOST_WORKFLOWS" ] ; then
        HAVE_DIRS=0
        echo "The path to workflows is not set or can not be found."
        echo "Workflows are expected here: $OLIVE_HOST_WORKFLOWS"
    fi
    if [ ! -d "$OLIVE_HOST_DOCSITE" ] ; then
        HAVE_DIRS=0
        echo "Can not find documentation."
        echo "It is expected here: $OLIVE_HOST_DOCSITE"
    fi
    if [ $HAVE_DIRS -eq 0 ] ; then
        echo "Because of missing directories, can not start the container."
        if [ ! -d "$MY_OLIVE_APP_DATA" ] ; then
            echo "It looks like you do not have the OLIVE_APP_DATA environment"
            echo "variable set, and there is no ../oliveAppData directory here."
            echo "So therefore, plugins, workflows, and server data can not be found."
        fi
        exit 5
    fi

    MARTINI_SERVER_ARGS=${MARTINI_SERVER_ARGS}
    NGINX_CONF="nginx.no_ssl.conf"
    DOCKER_GPU_ARG=""
    HTTP_PROTOCOL="http"
    OLIVE_SERVER_USE_SECURE_PORTS=false
    OLIVE_SERVER_USE_GPU=false

    for i in ${@:2}
    do
      case $i in 

        "--tls_server_only")
          NGINX_CONF="nginx.ssl.conf"
          HTTP_PROTOCOL="https"
          OLIVE_SERVER_USE_SECURE_PORTS=true
          ;;
        
        "--tls_server_and_client")
          NGINX_CONF="nginx.ssl.client.conf"
          HTTP_PROTOCOL="https"
          OLIVE_SERVER_USE_SECURE_PORTS=true
          ;;
        
        "--gpu")
          DOCKER_GPU_ARG="--gpus all"
          OLIVE_SERVER_USE_GPU=true
          ;;
        
        *)
          MARTINI_SERVER_ARGS="${MARTINI_SERVER_ARGS} $i"
          ;;
      esac
    done    

    # the oliveAppData/certs dir needs to exist regardless of whether TLS is being used so proactively create it
    # eventually this should be moved to an 'install' command if/when one is created
    if [[ ! -d $OLIVE_HOST_CERT_FILES ]]; then
      mkdir -p $OLIVE_HOST_CERT_FILES
    fi

    if [[ $NGINX_CONF == "nginx.ssl.conf" || $NGINX_CONF == "nginx.ssl.client.conf" ]]; then
      if [[ ! -f "$OLIVE_HOST_CERT_FILES/server.crt" ]]; then
        echo "Error: A server certificate must exist at $OLIVE_HOST_CERT_FILES/server.crt when starting with a TLS flag!"
        echo "Please copy server certificate to $OLIVE_HOST_CERT_FILES/server.crt and try starting again."
        exit 5
      fi

      if [[ ! -f "$OLIVE_HOST_CERT_FILES/server.key" ]]; then
        echo "Error: A server certificate key must exist at $OLIVE_HOST_CERT_FILES/server.key when starting with a TLS flag!"
        echo "Please copy server certificate key to $OLIVE_HOST_CERT_FILES/server.key and try starting again."
        exit 5
      fi

      if [[ $NGINX_CONF == "nginx.ssl.client.conf" && ! -f "$OLIVE_HOST_CERT_FILES/clientCA.crt" ]]; then
        echo "Error: A certificate authority to use for validating client certificates must exist at $OLIVE_HOST_CERT_FILES/clientCA.crt when starting with '--tls_server_and_client' flag!"
        echo "Please copy certificate authority to $OLIVE_HOST_CERT_FILES/clientCA.crt and try starting again."
        exit 5
      fi

      if [[ ! -f "$OLIVE_HOST_CERT_FILES/passwords.txt" ]]; then
        echo "Error: A 'passwords.txt' file is required at $OLIVE_HOST_CERT_FILES/passwords.txt when starting with a TLS flag!"
        echo "'passwords.txt' should contain passphrases for secret keys where each passphrase is specified on a separate line. Passphrases are tried in turn when loading the key."
        echo "**Note**: The file can be empty if passhprases are not required."
        echo "Please create $OLIVE_HOST_CERT_FILES/passwords.txt and try starting again."
        exit 5
      fi
    fi

    # the oliveAppData/olive.conf file should be created if one doesn't already exist
    if [[ ! -f $OLIVE_HOST_OLIVE_CONF_FILE ]]; then
      touch $OLIVE_HOST_OLIVE_CONF_FILE
    fi

    MARTINI_OLIVE_DATA="-v $MY_OLIVE_APP_DATA/olive-data:/olive-data"
    if [ "$LOCAL_OLIVE_DATA" != "" ] ; then
        MARTINI_OLIVE_DATA="-v $LOCAL_OLIVE_DATA:/olive-data"
    fi

    OLIVE_SERVER_PORTS="-p $OLIVE_HOST_PORT_OLIVESERVER:5588 -p $OLIVE_HOST_PORT_OLIVESECOND:5589 -p $OLIVE_HOST_PORT_OLIVESTREAMING:5590 -p $OLIVE_HOST_PORT_OLIVESTREAMINGTWO:5591"
    if [ "$OLIVE_SERVER_USE_SECURE_PORTS" = true ]; then
      OLIVE_SERVER_PORTS="-p $OLIVE_HOST_PORT_OLIVESERVER:5006"
    fi

    set -x
    docker run  --detach \
                --entrypoint bash \
                --env MARTINI_SERVER_ARGS="$MARTINI_SERVER_ARGS" \
                --env NGINX_CONF="$NGINX_CONF" \
                --env START_OLIVE_WEBSOCKET=$OLIVE_SERVER_USE_SECURE_PORTS \
                --env GPU_SUPPORT_ENABLED=$OLIVE_SERVER_USE_GPU \
                ${DOCKER_GPU_ARG} \
                --name olive-martini-$TAG \
                ${OLIVE_SERVER_PORTS} \
                -p $OLIVE_HOST_PORT_TEST:5004 \
                -p $OLIVE_HOST_PORT_DOCSERVER:8070 \
                -p $OLIVE_HOST_PORT_WEBSERVER:8080 \
                --rm=true  \
                --shm-size="8000M" \
                -v $OLIVE_HOST_CERT_FILES:/etc/nginx/certs \
                -v $OLIVE_HOST_PLUGINS:/home/olive/olive/plugins \
                -v $OLIVE_HOST_SERVER:/home/olive/olive/server \
                -v $OLIVE_HOST_OLIVE_CONF_FILE:/home/olive/olive/olive.conf \
                -v $OLIVE_HOST_WORKFLOWS:/opt/olive-broker/data/workflows \
                -v $OLIVE_HOST_DOCSITE:/var/www/html/help \
                $MARTINI_OLIVE_DATA \
                olive-martini:$TAG \
                /opt/olive/martini-process.sh
    set +x
    if [ "$DELAY" != "" ] ; then
        sleep $DELAY
    fi
    RUNNING_CONTAINER=$(docker ps -a | grep olive-martini:$TAG | cut -f 1 -d ' ')
    if [ "$RUNNING_CONTAINER" = "" ] ; then
        echo "Container did not start up."
        exit 2
    fi
    UNAME=$(uname -n)
    cat <<-EOF
	Started the container.
	From this machine:
	    Run Nightingale (Olive GUI) using server localhost and port $OLIVE_HOST_PORT_OLIVESERVER.
	    Use a web browser to $HTTP_PROTOCOL://localhost:$OLIVE_HOST_PORT_WEBSERVER/help/ to see the documentation.
	    Use a web browser to $HTTP_PROTOCOL://localhost:$OLIVE_HOST_PORT_WEBSERVER to use the Raven UI

	From any other machine:
	    Run Nightingale (Olive GUI) using server $UNAME and port $OLIVE_HOST_PORT_OLIVESERVER.
	    Use a web browser to $HTTP_PROTOCOL://$UNAME:$OLIVE_HOST_PORT_WEBSERVER/help/ to see the documentation.
	    Use a web browser to $HTTP_PROTOCOL://$UNAME:$OLIVE_HOST_PORT_WEBSERVER to use the Raven UI.

	Installed plugins (mounted from $OLIVE_HOST_PLUGINS) are:
	EOF

    MARTINI_CONTAINER_NAME=$(docker ps -a | grep olive-martini:$TAG | cut -f 1 -d ' ')
    docker exec -it "$MARTINI_CONTAINER_NAME"  /bin/bash -c 'ls /home/olive/olive/plugins'
    echo ""
    echo "Installed workflows (mounted from $OLIVE_HOST_WORKFLOWS) are:"
    docker exec -it "$MARTINI_CONTAINER_NAME" /bin/bash -c 'ls /opt/olive-broker/data/workflows/*.{json,workflow} 2>/dev/null'
    echo ""
    if [[ $NGINX_CONF == "nginx.ssl.conf" || $NGINX_CONF == "nginx.ssl.client.conf" ]]; then
      echo "Certificate files (mounted from $OLIVE_HOST_CERT_FILES) are:"
      docker exec -it "$MARTINI_CONTAINER_NAME" /bin/bash  -c  'ls /etc/nginx/certs'
      echo ""
    fi
    ;;

  status)
    RAN_CMD="true"
    if [ "$RUNNING_CONTAINER" = "" ] ; then
        echo "No Martini container is running so can not run the $1 command."
        exit 2
    fi
    docker exec -it "$RUNNING_CONTAINER" /bin/bash -c /opt/olive/martini-status.sh
    ;;

  stop)
    RAN_CMD="true"
    if [ "$RUNNING_CONTAINER" = "" ] ; then
        echo "No Martini container is running so can not run the $1 command."
        exit 2
    fi
    #FULLNAME=olive-martini-$TAG
    #MINE=`docker ps | grep $FULLNAME |awk '{print $1;}'`

    #if [ "$MINE" != "" ] ; then
        docker stop "$RUNNING_CONTAINER"
        # The next line should not be needed because stop leads to rm.
        # docker rm $MINE
    #else
    #    echo No $FULLNAME docker running.
    #fi
    ;;

  cli)
    RAN_CMD="true"
    if [ "$RUNNING_CONTAINER" = "" ] ; then
        echo "No Martini container is running so can not run the $1 command."
        exit 2
    fi
    docker exec -it "$RUNNING_CONTAINER" /bin/bash
    ;;

  net)
    RAN_CMD="true"
    if [ "$RUNNING_CONTAINER" = "" ] ; then
        echo "No Martini container is running so can not run the $1 command."
        exit 2
    fi
    netstat -l | grep ":5[05][0-9][0-9] "
    ;;

  version)
    RAN_CMD="true"
    echo "Martini vAIWDP-beta.rc3 Olive v6.0.0  Broker v2.0.1  Raven v2.0.2  Secure Websockets v0.1.2"
    ;;

esac

if [ "$RAN_CMD" = "false" ] ; then
    echo "The $1 command is not supported."
    echo "Supported commands include: cli, list, log, start, stop, status, net, version."
fi
