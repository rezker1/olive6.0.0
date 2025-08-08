@REM ----------------------------------------------------------------------------
@REM  Copyright 2001-2006 The Apache Software Foundation.
@REM
@REM  Licensed under the Apache License, Version 2.0 (the "License");
@REM  you may not use this file except in compliance with the License.
@REM  You may obtain a copy of the License at
@REM
@REM       http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM  Unless required by applicable law or agreed to in writing, software
@REM  distributed under the License is distributed on an "AS IS" BASIS,
@REM  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM  See the License for the specific language governing permissions and
@REM  limitations under the License.
@REM ----------------------------------------------------------------------------
@REM
@REM   Copyright (c) 2001-2006 The Apache Software Foundation.  All rights
@REM   reserved.

@echo off

set ERROR_CODE=0

:init
@REM Decide how to startup depending on the version of windows

@REM -- Win98ME
if NOT "%OS%"=="Windows_NT" goto Win9xArg

@REM set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" @setlocal

@REM -- 4NT shell
if "%eval[2+2]" == "4" goto 4NTArgs

@REM -- Regular WinNT shell
set CMD_LINE_ARGS=%*
goto WinNTGetScriptDir

@REM The 4NT Shell from jp software
:4NTArgs
set CMD_LINE_ARGS=%$
goto WinNTGetScriptDir

:Win9xArg
@REM Slurp the command line arguments.  This loop allows for an unlimited number
@REM of arguments (up to the command line limit, anyway).
set CMD_LINE_ARGS=
:Win9xApp
if %1a==a goto Win9xGetScriptDir
set CMD_LINE_ARGS=%CMD_LINE_ARGS% %1
shift
goto Win9xApp

:Win9xGetScriptDir
set SAVEDIR=%CD%
%0\
cd %0\..\.. 
set BASEDIR=%CD%
cd %SAVEDIR%
set SAVE_DIR=
goto repoSetup

:WinNTGetScriptDir
set BASEDIR=%~dp0\..

:repoSetup
set REPO=


if "%JAVACMD%"=="" set JAVACMD=java

if "%REPO%"=="" set REPO=%BASEDIR%\repo

set CLASSPATH="%BASEDIR%"\etc;"%REPO%"\com\google\protobuf\protobuf-java\3.25.5\protobuf-java-3.25.5.jar;"%REPO%"\com\google\protobuf\protobuf-java-util\3.25.5\protobuf-java-util-3.25.5.jar;"%REPO%"\com\google\code\findbugs\jsr305\3.0.2\jsr305-3.0.2.jar;"%REPO%"\com\google\code\gson\gson\2.8.9\gson-2.8.9.jar;"%REPO%"\com\google\errorprone\error_prone_annotations\2.18.0\error_prone_annotations-2.18.0.jar;"%REPO%"\com\google\guava\guava\32.0.1-jre\guava-32.0.1-jre.jar;"%REPO%"\com\google\guava\failureaccess\1.0.1\failureaccess-1.0.1.jar;"%REPO%"\com\google\guava\listenablefuture\9999.0-empty-to-avoid-conflict-with-guava\listenablefuture-9999.0-empty-to-avoid-conflict-with-guava.jar;"%REPO%"\org\checkerframework\checker-qual\3.33.0\checker-qual-3.33.0.jar;"%REPO%"\com\google\j2objc\j2objc-annotations\2.8\j2objc-annotations-2.8.jar;"%REPO%"\com\googlecode\json-simple\json-simple\1.1.1\json-simple-1.1.1.jar;"%REPO%"\junit\junit\4.10\junit-4.10.jar;"%REPO%"\org\hamcrest\hamcrest-core\1.1\hamcrest-core-1.1.jar;"%REPO%"\org\json\json\20240303\json-20240303.jar;"%REPO%"\org\zeromq\jeromq\0.5.2\jeromq-0.5.2.jar;"%REPO%"\eu\neilalexander\jnacl\1.0.0\jnacl-1.0.0.jar;"%REPO%"\org\slf4j\slf4j-api\2.0.13\slf4j-api-2.0.13.jar;"%REPO%"\ch\qos\logback\logback-core\1.5.6\logback-core-1.5.6.jar;"%REPO%"\ch\qos\logback\logback-classic\1.5.6\logback-classic-1.5.6.jar;"%REPO%"\commons-lang\commons-lang\2.6\commons-lang-2.6.jar;"%REPO%"\commons-io\commons-io\2.4\commons-io-2.4.jar;"%REPO%"\commons-cli\commons-cli\1.4\commons-cli-1.4.jar;"%REPO%"\com\sri\speech\olive\api\olive-api\6.0.0\olive-api-6.0.0.jar

set ENDORSED_DIR=
if NOT "%ENDORSED_DIR%" == "" set CLASSPATH="%BASEDIR%"\%ENDORSED_DIR%\*;%CLASSPATH%

if NOT "%CLASSPATH_PREFIX%" == "" set CLASSPATH=%CLASSPATH_PREFIX%;%CLASSPATH%

@REM Reaching here means variables are defined and arguments have been captured
:endInit

%JAVACMD% %JAVA_OPTS% -Xms4096m -classpath %CLASSPATH% -Dapp.name="OliveWorkflow" -Dapp.repo="%REPO%" -Dapp.home="%BASEDIR%" -Dbasedir="%BASEDIR%" com.sri.speech.olive.api.client.OliveWorkflow %CMD_LINE_ARGS%
if %ERRORLEVEL% NEQ 0 goto error
goto end

:error
if "%OS%"=="Windows_NT" @endlocal
set ERROR_CODE=%ERRORLEVEL%

:end
@REM set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" goto endNT

@REM For old DOS remove the set variables from ENV - we assume they were not set
@REM before we started - at least we don't leave any baggage around
set CMD_LINE_ARGS=
goto postExec

:endNT
@REM If error code is set to 1 then the endlocal was done already in :error.
if %ERROR_CODE% EQU 0 @endlocal


:postExec

if "%FORCE_EXIT_ON_ERROR%" == "on" (
  if %ERROR_CODE% NEQ 0 exit %ERROR_CODE%
)

exit /B %ERROR_CODE%
