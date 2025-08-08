
 param (
    [string]$command = "help",
    [string]$subcommand = "",
    [switch]$debug = $false,
    [switch]$gpu = $false,
    [switch]$tls_server_only = $false,
    [switch]$tls_server_and_client = $false
 )

$TAG = "aiwdp-beta.rc3"
$FULLNAME = "martini-$TAG"
$DELAY = 7

if ($env:OLIVE_APP_DATA -eq $null) {
    $MY_OLIVE_APP_DATA = [System.IO.Path]::GetFullPath("..\oliveAppData")
} else {
    $MY_OLIVE_APP_DATA = [System.IO.Path]::GetFullPath($env:OLIVE_APP_DATA)
}

if ($env:OLIVE_HOST_PLUGINS -eq $null) {
    $OLIVE_HOST_PLUGINS = $MY_OLIVE_APP_DATA+"\plugins"
} else {
    $OLIVE_HOST_PLUGINS = [System.IO.Path]::GetFullPath($env:OLIVE_HOST_PLUGINS)
}

if ($env:OLIVE_HOST_WORKFLOWS -eq $null) {
    $OLIVE_HOST_WORKFLOWS = $MY_OLIVE_APP_DATA+"\workflows"
} else {
    $OLIVE_HOST_WORKFLOWS = [System.IO.Path]::GetFullPath($env:OLIVE_HOST_WORKFLOWS)
}

if ($env:OLIVE_HOST_CERT_FILES -eq $null) {
    $OLIVE_HOST_CERT_FILES = $MY_OLIVE_APP_DATA+"\certs"
} else {
    $OLIVE_HOST_CERT_FILES = [System.IO.Path]::GetFullPath($env:OLIVE_HOST_CERT_FILES)
}

if ($env:OLIVE_HOST_DOCSITE -eq $null) {
    $OLIVE_HOST_DOCSITE = [System.IO.Path]::GetFullPath("..\docs")
} else {
    $OLIVE_HOST_DOCSITE = [System.IO.Path]::GetFullPath($env:OLIVE_HOST_DOCSITE)
}

if ($env:OLIVE_HOST_SERVER -eq $null) {
    $OLIVE_HOST_SERVER = $MY_OLIVE_APP_DATA+"\server"
} else {
    $OLIVE_HOST_SERVER = [System.IO.Path]::GetFullPath($env:OLIVE_HOST_SERVER)
}

if ($OLIVE_HOST_PORT_OLIVESERVER -eq $null) {
    if ($env:OLIVE_HOST_PORT_OLIVESERVER -eq $null) {
        $OLIVE_HOST_PORT_OLIVESERVER = 5588
    } else {
        $OLIVE_HOST_PORT_OLIVESERVER = $env:OLIVE_HOST_PORT_OLIVESERVER
    }
}

if ($OLIVE_HOST_PORT_OLIVESECOND -eq $null) {
    if ($env:OLIVE_HOST_PORT_OLIVESECOND -eq $null) {
        $OLIVE_HOST_PORT_OLIVESECOND = 5589
    } else {
        $OLIVE_HOST_PORT_OLIVESECOND = $env:OLIVE_HOST_PORT_OLIVESECOND
    }
}

if ($OLIVE_HOST_PORT_OLIVESTREAMING -eq $null) {
    if ($env:OLIVE_HOST_PORT_OLIVESTREAMING -eq $null) {
        $OLIVE_HOST_PORT_OLIVESTREAMING = 5590
    } else {
        $OLIVE_HOST_PORT_OLIVESTREAMING = $env:OLIVE_HOST_PORT_OLIVESTREAMING
    }
}

if ($OLIVE_HOST_PORT_OLIVESTREAMINGTWO -eq $null) {
    if ($env:OLIVE_HOST_PORT_OLIVESTREAMINGTWO -eq $null) {
        $OLIVE_HOST_PORT_OLIVESTREAMINGTWO = 5591
    } else {
        $OLIVE_HOST_PORT_OLIVESTREAMINGTWO = $env:OLIVE_HOST_PORT_OLIVESTREAMINGTWO
    }
}

if ($OLIVE_HOST_PORT_TEST -eq $null) {
    if ($env:OLIVE_HOST_PORT_TEST -eq $null) {
        $OLIVE_HOST_PORT_TEST = 5004
    } else {
        $OLIVE_HOST_PORT_TEST = $env:OLIVE_HOST_PORT_TEST
    }
}

if ($OLIVE_HOST_PORT_DOCSERVER -eq $null) {
    if ($env:OLIVE_HOST_PORT_DOCSERVER -eq $null) {
        $OLIVE_HOST_PORT_DOCSERVER = 5570
    } else {
        $OLIVE_HOST_PORT_DOCSERVER = $env:OLIVE_HOST_PORT_DOCSERVER
    }
}

if ($OLIVE_PORT_WEBSERVER -eq $null) {
    if ($env:OLIVE_PORT_WEBSERVER -eq $null) {
        $OLIVE_HOST_PORT_WEBSERVER = 5580
    } else {
        $OLIVE_HOST_PORT_WEBSERVER = $env:OLIVE_HOST_PORT_WEBSERVER
    }
}

$OLIVE_HOST_OLIVE_CONF_FILE = $MY_OLIVE_APP_DATA+"\olive.conf"

# RUNNING_CONTAINER will be "" if no container is running.
$RUNNING = (docker ps | findstr -i martini)
if ($RUNNING -ne $null) {
    $RUNNING_CONTAINER = (docker ps | findstr -i martini).split()[0]
}

switch ($command) {
    "help" {
    Write-Host "
	Supported commands include: cli, help, list, log, start, stop, status, net, and version.
	martini.bat version: Prints out version information.
        martini.bat start: Start the container.
            Optional flags:
              -gpu: Enable Martini access to GPUs, if available. Be sure to properly configure any plugin/domains to be used with GPU devices as outlined in the documentation.
              -tls_server_only: The server will be configured with a certificate and will only respond to HTTPS requests; clients aren't required to send certificate but must use HTTPS protocol (one-way TLS)
              -tls_server_and_client: Both the server and clients will need to communicate with certificates over HTTPS (two-way TLS)
              -debug: Activate 'debug' mode for the OLIVE Server. All server logs will be maintained to aid in troubleshooting and debugging.
        martini.bat stop: Stop the container.
        martini.bat list: List the running container.
        martini.bat log: print out a log file.  Options are:
	    raven, martini, message, server, pa, pe, po
        martini.bat status: Shows status of the processes on the container.
        martini.bat net: Shows the ports on the host that the container is listening on.
        martini.bat cli: Starts a shell on the container for debugging.
    "
    }
    "cli" {
        if ($RUNNING -eq $null) {
            Write-Host "No container is running, and one is required for the %1 command."
            exit
        }
        docker exec -it $RUNNING_CONTAINER /bin/bash
    }
    "list" {
        docker ps | Select -First 1
        docker ps | Select-String $FULLNAME
    }
    "load" {
        Write-Host "Loading the martini container which will take awhile."
        docker load --input martini-container.tar
    }
    "log" {
        if ($RUNNING -eq $null) {
            Write-Host "No container is running, and one is required for the %1 command."
            exit
        }
        $LOGNAME = "/home/olive/olive/oliveserver.log"
        switch -Wildcard ($subcommand) {
            "*pa*" { $LOGNAME = "/home/olive/olive/nginx_access.log"}
            "*pe*" { $LOGNAME = "/home/olive/olive/nginx_error.log"}
            "*po*" { $LOGNAME = "/home/olive/olive/nginx.log"}
            "*raven*" { $LOGNAME = "/home/olive/olive/httpdserver.log" }
            "*martini" { $LOGNAME = "/home/olive/olive/martini-process.log" }
            "*mess*" { $LOGNAME = "/home/olive/olive/messagebroker.log" }
            "*server" { $LOGNAME = "/home/olive/olive/oliveserver.log" }
        }
        docker exec -it $RUNNING_CONTAINER tail $args $LOGNAME -n +0
    }
    "start" {
        $HAVE_DIRS = 1
        if (-not (Test-Path $OLIVE_HOST_PLUGINS)) {
            $HAVE_DIRS = 0
            Write-Host "The path to plugins is not set or can not be found."
            Write-Host "This is done via `$OLIVE_HOST_PLUGINS which is set to $OLIVE_HOST_PLUGINS now."
        }
        if (-not (Test-Path $OLIVE_HOST_WORKFLOWS)) {
            $HAVE_DIRS = 0
            Write-Host "The path to workflows is not set or can not be found."
            Write-Host "This is done via `$OLIVE_HOST_WORKFLOWS which is set to $OLIVE_HOST_WORKFLOWS now."
        }
        if (-not (Test-Path $OLIVE_HOST_DOCSITE)) {
            $HAVE_DIRS = 0
            Write-Host "The path to documentation is not set or can not be found."
            Write-Host "This is done via `$OLIVE_HOST_DOCSITE which is set to $OLIVE_HOST_DOCSITE now."
        }
        if ($HAVE_DIRS -eq 0) {
            Write-Host "Because of missing directories, can not start the container."
            exit 5
        }
        $NGINX_CONF="nginx.no_ssl.conf"
        $HTTP_PROTOCOL="http"
        $OLIVE_SERVER_USE_SECURE_PORTS=$false

        if ($tls_server_only -or $tls_server_and_client) {
            if ($tls_server_and_client) {
                $NGINX_CONF="nginx.ssl.client.conf"
            } elseif ($tls_server_only){
                $NGINX_CONF="nginx.ssl.conf"
            }
            $HTTP_PROTOCOL="https"
            $OLIVE_SERVER_USE_SECURE_PORTS=$true

            $MARTINI_SERVER_ARGS = "$env:MARTINI_SERVER_ARGS $args"
        } else {
            $MARTINI_SERVER_ARGS = "$env:MARTINI_SERVER_ARGS $subcommand $args"
        }

        # the oliveAppData/certs dir needs to exist regardless of whether TLS is being used so proactively create it
        # eventually this should be moved to an 'install' command if/when one is created
        if (-not (Test-Path $OLIVE_HOST_CERT_FILES)) {
            mkdir $OLIVE_HOST_CERT_FILES > $null
        }

        if ($NGINX_CONF -eq "nginx.ssl.conf" -or $NGINX_CONF -eq "nginx.ssl.client.conf") {
            if (-not (Test-Path $OLIVE_HOST_CERT_FILES/server.crt)) {
                Write-Host "Error: A server certificate must exist at ${OLIVE_HOST_CERT_FILES}\server.crt when starting with a TLS flag!"
                Write-Host "Please copy server certificate to ${OLIVE_HOST_CERT_FILES}\server.crt and try starting again."
                exit 5
            }

            if (-not (Test-Path $OLIVE_HOST_CERT_FILES/server.key)) {
                Write-Host "Error: A server certificate key must exist at ${OLIVE_HOST_CERT_FILES}\server.key when starting with a TLS flag!"
                Write-Host "Please copy server certificate key to ${OLIVE_HOST_CERT_FILES}\server.key and try starting again."
                exit 5
            }

            if ($NGINX_CONF -eq "nginx.ssl.client.conf" -and (-not (Test-Path $OLIVE_HOST_CERT_FILES/clientCA.crt))) {
                Write-Host "Error: A certificate authority to use for validating client certificates must exist at ${OLIVE_HOST_CERT_FILES}\clientCA.crt when starting with '-tls_server_and_client' flag!"
                Write-Host "Please copy certificate authority to ${OLIVE_HOST_CERT_FILES}\clientCA.crt and try starting again."
                exit 5
            }

            if (-not (Test-Path $OLIVE_HOST_CERT_FILES/passwords.txt)) {
                Write-Host "Error: A 'passwords.txt' file is required at ${OLIVE_HOST_CERT_FILES}\passwords.txt when starting with a TLS flag!"
                Write-Host "'passwords.txt' should contain passphrases for secret keys where each passphrase is specified on a separate line. Passphrases are tried in turn when loading the key."
                Write-Host "**Note**: The file can be empty if passhprases are not required."
                Write-Host "Please create ${OLIVE_HOST_CERT_FILES}\passwords.txt and try starting again."
                exit 5
            }
        }

        if ($debug) {
           $MARTINI_SERVER_ARGS = "--debug $MARTINI_SERVER_ARGS"
        }

        # the oliveAppData/olive.conf file should be created if one doesn't already exist
        if (-not (Test-Path $OLIVE_HOST_OLIVE_CONF_FILE)) {
            New-Item $OLIVE_HOST_OLIVE_CONF_FILE | Out-Null
        }

        $MARTINI_OLIVE_DATA = "-v "+$MY_OLIVE_APP_DATA+"/olive-data:/olive-data"
        if ($env:LOCAL_OLIVE_DATA -ne $null) {
            $MARTINI_OLIVE_DATA = "-v "+$env:LOCAL_OLIVE_DATA+":/olive-data"
        }

        $OLIVE_SERVER_PORTS = "-p ${OLIVE_HOST_PORT_OLIVESERVER}:5588 -p ${OLIVE_HOST_PORT_OLIVESECOND}:5589 -p ${OLIVE_HOST_PORT_OLIVESTREAMING}:5590 -p ${OLIVE_HOST_PORT_OLIVESTREAMINGTWO}:5591"
        if ($OLIVE_SERVER_USE_SECURE_PORTS) {
            $OLIVE_SERVER_PORTS="-p ${OLIVE_HOST_PORT_OLIVESERVER}:5006"
            $OLIVE_SERVER_USE_SECURE_PORTS="true"
        }

        # Next line is for debugging
        # Set-PSDebug -Trace 1

        $DOCKER_GPU_ARG = if ($gpu) { "--gpus all"} else {""}
        $OLIVE_SERVER_USE_GPU = if ($gpu) { "true" } else { "false" }

        # This command needs to be on one line.
        Start-Process -NoNewWindow docker -ArgumentList "run --entrypoint bash ${DOCKER_GPU_ARG}     --shm-size=8000M     --env MARTINI_SERVER_ARGS=""$MARTINI_SERVER_ARGS""              --env NGINX_CONF=""$NGINX_CONF""             --env START_OLIVE_WEBSOCKET=""$OLIVE_SERVER_USE_SECURE_PORTS""             --env GPU_SUPPORT_ENABLED=""$OLIVE_SERVER_USE_GPU""             $OLIVE_SERVER_PORTS             -p ${OLIVE_HOST_PORT_TEST}:5004              -p ${OLIVE_HOST_PORT_DOCSERVER}:8070             -p ${OLIVE_HOST_PORT_WEBSERVER}:8080               --name martini-$TAG             --rm              -v ${OLIVE_HOST_CERT_FILES}:/etc/nginx/certs              -v ${OLIVE_HOST_PLUGINS}:/home/olive/olive/plugins                -v ${OLIVE_HOST_SERVER}:/home/olive/olive/server               -v ${OLIVE_HOST_OLIVE_CONF_FILE}:/home/olive/olive/olive.conf               -v ${OLIVE_HOST_WORKFLOWS}:/opt/olive-broker/data/workflows               -v ${OLIVE_HOST_DOCSITE}:/var/www/html/help             ${MARTINI_OLIVE_DATA}             olive-martini:${TAG}        /opt/olive/martini-process.sh"
        if ($DELAY -ne $null) {
            Start-Sleep $DELAY
        }
        $UNAME = (hostname)
        Write-Host "	Started the container.
	From this machine:
	    Run Nightingale (Olive GUI) using server localhost and port $OLIVE_HOST_PORT_OLIVESERVER.
	    Use a web browser to ${HTTP_PROTOCOL}://localhost:$OLIVE_HOST_PORT_WEBSERVER/help/ to see the documentation.
	    Use a web browser to ${HTTP_PROTOCOL}://localhost:$OLIVE_HOST_PORT_WEBSERVER to use the Web UI.

	From any other machine:
	    Run Nightingale (Olive GUI) using server ${UNAME} and port $OLIVE_HOST_PORT_OLIVESERVER.
	    Use a web browser to ${HTTP_PROTOCOL}://${UNAME}:$OLIVE_HOST_PORT_WEBSERVER/help/ to see the documentation.
	    Use a web browser to ${HTTP_PROTOCOL}://${UNAME}:$OLIVE_HOST_PORT_WEBSERVER to use the Web UI.

	Installed plugins (mounted from $OLIVE_HOST_PLUGINS) are:"
        $RUNNING = (docker ps | findstr -i martini)
        if ($RUNNING -ne $null) {
            $RUNNING_CONTAINER = (docker ps | findstr -i martini).split()[0]
        }
        docker exec -it $RUNNING_CONTAINER /bin/bash -c "ls /home/olive/olive/plugins"
        Write-Host ""
        Write-Host "Installed workflows (mounted from $OLIVE_HOST_WORKFLOWS) are:"
        docker exec -it $RUNNING_CONTAINER /bin/bash -c "ls /opt/olive-broker/data/workflows"
        Write-Host ""

        if ($NGINX_CONF -eq "nginx.ssl.conf" -or $NGINX_CONF -eq "nginx.ssl.client.conf") {
            Write-Host ""
            Write-Host "Certificate files (mounted from $OLIVE_HOST_CERT_FILES) are:"
            docker exec -it $RUNNING_CONTAINER /bin/bash -c "ls /etc/nginx/certs"
            Write-Host ""
        }
    }
    "status" {
        if ($RUNNING -eq $null) {
            Write-Host "No container is running, and one is required for the %1 command."
            exit
        }
        docker exec -it $RUNNING_CONTAINER /bin/bash -c /opt/olive/martini-status.sh
    }
    "stop" {
        if ($RUNNING -eq $null) {
            Write-Host "No container is running, and one is required for the %1 command."
            exit
        }
        docker stop $RUNNING_CONTAINER
    }
    "net" {
        if ($RUNNING -eq $null) {
            Write-Host "No container is running, and one is required for the %1 command."
            exit
        }
        netstat -a -n | findstr /R /C:":5[05][0-9][0-9][ ]"
    }
    "version" {
        Write-Host "Martini vAIWDP-beta.rc3 Olive v6.0.0  Broker v2.0.1  Raven v2.0.2  Secure Websockets v0.1.2"
    }
    default {
        Write-Host "The $command command is not supported.
                    Supported commands include: cli, list, start, stop, status, net, version."
    }
}
