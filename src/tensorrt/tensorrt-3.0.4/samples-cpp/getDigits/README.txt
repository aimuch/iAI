getDigits HOWTO
================================================================================

	These directions will walk through the steps required to retrieve files from
a digits server and execute them utilizing TensorRT's giexec tool. The
assumptions that this tutorial makes are as follows.
   * The user has an IP address and port of a server running digits.
   * The digits server has a model that has been trained for at least one epoch.
   * The user has installed and built the giexec tool.
   * The user has successfully verified TensorRT works on the device by running
   	 one of the samples.
   * unzip, tar, gunzip and bunzip2 tools are installed.


   There are four steps to the process of extracting a network model from digits
and deploying it utilizing TensorRT. Download, extract, generate, and deploy.
Lines of '+' characters dillineate shell interations and Lines of '=' characters
dilleniate different sections.


Download
================================================================================

	The script 'download-digits-model.py' can be used to download the network
model in a compressed format from a digits installation. In order to download
the model, there is one required and two optional arguments to the script.
These options are: 
    * File - Required - The name of the compressed file to store results in.
	  Must end in .zip, .tar, .tar.gz, or .tar.bz2.
	* Hostname - Optional - The IPv4 address of the server that is hosting
	  digits. Uses localhost if not specified. Command line option is either
	  '-n' or '--hostname'.
	* Port - Optional - The port on the server to connect to digits. Uses port
	  80 if not specified.  Command line option is either '-p' or '--port'.

	The format of the command is:
'download-digits-model.py [-p|--port <Port>] [-n|--hostname <Hostname>] <File>.'
	The '<....>' tags represent items that must be supplied by the user. The
'[...]' tags are the optional commands. The '|' tag means that one or the other
is used, not both.


	An example run is as follows:

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
$ python download-digits-model.py  test.zip
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	This launches the script to connect to the local installation of digits and
stores the results in test.zip

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[Num] Job ID               Status     Name                
----------------------------------------------------------
[  1] 20160606-165113-de6e Done       GoogLeNet_test      
Select a job
>>> 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	On this page, we see that there is a single job available from the digits
server. Selecting '1' and pressing enter will provide the user with the option
to select which epoch of this run they wish to download into 'test.zip'.

The results of this command is as follows:

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[Num] Epoch     
----------------
[  1] 1         
[  2] 2         
[  3] 3         
[  4] 4         
[  5] 5         
[  6] 6         
[  7] 7         
[  8] 8         
[  9] 9         
[ 10] 10        
[ 11] 11        
[ 12] 12        
[ 13] 13        
[ 14] 14        
[ 15] 15        
[ 16] 16        
[ 17] 17        
[ 18] 18        
[ 19] 19        
[ 20] 20        
[ 21] 21        
[ 22] 22        
[ 23] 23        
[ 24] 24        
[ 25] 25        
[ 26] 26        
[ 27] 27        
[ 28] 28        
[ 29] 29        
[ 30] 30        
Select a snapshot (leave blank for default=30)
>>> 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	The user now must select their epoch they wish to download to their local
machine from the server for extraction. The data is download and stored in a
compressed format in the filename specified in the command line.


Extract
================================================================================

	Now that the file has been downloaded from the digits server, it needs to
be extracted in order to be used by giexec.

	If the filetype is .zip, the command is 'unzip <File>'.
	If the filetype is .tar, the command is 'tar xvf <File>'.
	If the filetype is .tar.gz, the command is 'tar zxvf <File>'.
	If the filetype is .tar.bz2, the command is 'tar jxvf <File>'.

	Once the files are extracted, then there is expected to be at least the
following files:
	* deploy.prototxt
	* train_val.prototxt
	* solver.prototxt
	* labels.txt
	* mean.binaryproto
	* snapshot_iter_#.caffemodel

	For running in giexec, deploy.prototxt and the caffemodel are needed.

Generate
================================================================================

	After building the samples that come with TensorRT, there will be a binary
in the bin/ directory called giexec. giexec is a tool to quickly utilize
TensorRT without having to execute code. In this section, the process for using
giexec to convert from a prototxt and caffemodel into a TensorRT serialized
engine will be described. For giexec, there are 4 options that will be utilized.
	* --deploy, used to specify the deploy.prototxt that was extracted.
	* --output, this specifies the output layers of the deploy.prototxt. There
	  can be more than one output layer, so specifying it more than once is ok.
	* --model, used to specify the caffemodel that was extracted.
	* --engine, used to specify the file to save the serialized TensorRT engine
	  for deployement.

	There are other options, but they are outside the scope of this task. An 
example execute would look like as follows:

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
>$./giexec --deploy=deploy.prototxt --output=prob --model=deploy.caffemodel \
					--engine=myengine.rt

deploy: deploy.prototxt
output: prob
model: deploy.caffemodel
engine: myengine.rt
Input "data": 3x224x224
Output "prob": 10x1x1
name=data, bindingIndex=0, buffers.size()=2
name=prob, bindingIndex=1, buffers.size()=2
Average over 10 runs is 1.17613 ms.
Average over 10 runs is 1.17624 ms.
Average over 10 runs is 1.1762 ms.
Average over 10 runs is 1.17618 ms.
Average over 10 runs is 1.17564 ms.
Average over 10 runs is 1.17339 ms.
Average over 10 runs is 1.1758 ms.
Average over 10 runs is 1.17527 ms.
Average over 10 runs is 1.17448 ms.
Average over 10 runs is 1.17494 ms.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	What we see is some information about the network being loaded and then some
timing information about this particular network being executed on the current
graphics device.

Deploy
================================================================================

	The serialized TensorRT engine that was produced can now be deployed onto the
the target device using the TensorRT deserialization API. See the samples for 
code on how to deserialize and execute an engine.
