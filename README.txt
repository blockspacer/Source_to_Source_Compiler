There are three folders,one Pdf file(Final Report.pdf),one shell script,CMakeLists.txt and one README.txt in my submission.

Descriptions of three folders
--------------------------------------
1>functions-rewrite - folder contains two files.
	CMakeLists.txt
	FunctionRewrite.cpp - This is my source code file.
2>Test_Programs -  This folder contains three test programs which will be used as input.
	cudaKernelCall.cu
	matrixAdd_org.cu
	matrixMul_org.cu
3>Expected_Output_Programs -  This folder contains three expected outputs of above mentioned three input programs.
	cudaKernelCall_smc.cu
	matrixAdd_smc.cu
	matrixMul_smc.cu


Tasks:-  I had following tasks to be implemented.
--------------------------------------
  
  1. add the following line to the beginning of  the CUDA file after all existing #include statements:

     #include "smc.h" 

  2. add the following line to the beginning of the definition of the kernel function:

     __SMC_Begin

  3. add the following line to the end of the definition of the kernel function:

     __SMC_End

  4. add the following arguments to the end of the argument list of the definition of the kernel function:

     dim3 __SMC_orgGridDim, int __SMC_workersNeeded, int *__SMC_workerCount, int * __SMC_newChunkSeq, int * __SMC_seqEnds

  5. in the definition of the kernel function, replace the references of blockIdx.x with

       (int)fmodf((float)__SMC_chunkID, (float)__SMC_orgGridDim.x);

     and replace the references of blockIdx.y with

       (int)(__SMC_chunkID/__SMC_orgGridDim.x);

  6. Replace the call of function grid(...) with 

       dim3 __SMC_orgGridDim(...)

     The arguments should be kept unchanged.

  7. add the following right before the call of the GPU kernel  function:

       __SMC_init();

  8. append the following arguments to the end of the GPU kernel function call:

       __SMC_orgGridDim, __SMC_workersNeeded, __SMC_workerCount, __SMC_newChunkSeq, __SMC_seqEnds

Implementation:-

--------------------------------------
I have successfully implemented the above mentioned eight code transformations. Here I would describe my approach of implementations and some assumptions which I made while implementation.I have first read the Clang AST documentation and learn about LibTooling, ASTMatcher and Rewriter. If we closely observe above mentioned eight modifications, then seven among them are related to code rewriting. The only one which is different is adding a new header file. Because we know, preprocessor directives and header files are not part of the clang AST. So, the handling of this modification differs from rest. For those seven similar kind of modifications, I have written a ASTMatcher for each one of them. ASTMatcher is like the query string based on which AST node of interest will be matched. Then, I have created child class inheriting MatchFinder parent class. Inside newly created class, virtual run method has been implemented. We know in C++, virtual function is such kind of function which is been declared in parent class and been defined in child classes. Inside this run method, the rewriting part has been done. 

1> Finding the declaration of CUDA kernel function - I have used CUDAGlobalAttr property of FunctionDecl class to identify the CUDA kernel function. 
2> Finding the CUDA kernel call - I have used cudaKernelCallExpr class to identify CUDA kernel call.
3> varDecl class was used to modify several variable declarations.
4> For the header file addition part, I have used PPCallbacks class. I have created a new class Find_Includes which inherits PPCallbacks class . Inside that class, I have implemented InclusionDirective() method. Inside this method, I have added the line "#include"smc.h". Then inside MyFrontendAction class which inherits ASTFrontendAction class, I have implemented BeginSourceFileAction() method. Inside that method, I have created a std::unique_ptr referencing Find_Includes class. This ptr has been passed as an argument for addPPCallbacks() method.

Instructions for copying my module in exact location :-
------------------------------------------------------------
According to the instruction given in moodle, the test machine has already been set up for cuda and ninja. So, I am specifying the instructions to execute my program only.
 
I have created "Copy.sh" file for copying my files in the exact location.If you run this file, it will copy my files in the appropriate location.
1> $ chmod a+x Copy.sh
2> $ ./Copy.sh

I have mentioned the absolute path in "Copy.sh". If it does not work, I am hereby giving the description of copying my files using linux terminal. I have given here the relative path instead of absolute path.
To make the following commands work, your current directory must be /home/ubuntu and you should have my submission folder "Jchakra" there.

1> copy the functions_rewrite folder in appropriate location. [Assumption is you are indide "Jchakra" folder]
	$ cp -r functions-rewrite ../llvm/llvm/tools/clang/tools/extra
2> copy CMakeLists.txt
	$ cp CMakeLists.txt ../llvm/llvm/tools/clang/tools/extra

Instructions for compilation  and execution :-
---------------------------------------------------------
To make the following instructions work, you need to copy Jchakra folder in /home/ubuntu location and your current directory should be same.
1> Open a new terminal.
2> $ cd llvm/build-release
3> $ ninja
4> bin/functions-rewrite ../../Jchakra/Test_Programs/matrixAdd_org.cu -- --cuda-host-only -I /usr/local/cuda/include   [ Change the path according to location of my "Test_Programs" in the system. The name of the input cuda file also needs to be changed everytime. The last part is an argument which depends upon the cuda path in the test machine.]
This should print the output in the console. You might get the error of some missing header files. You have to copy those header files in the /usr/local/cuda/include location. When I got the initial project folder, the header files were present there. I copied them in the exact location. 

Understanding the output :- 
--------------------------------------

I have implemented the eight modifications mentioned above. I have given expected output programs in the "Expected_Output_Programs" folder. The output of my program should match with those expected output programs. If you find difficulty reading the output program in console, you can create an output file also. Fo that, I have written a code in FunctionRewrite.cpp file ( The code is commemted out ). So you can easily modify it and generate output file.You can provide whatever name and location you want of the output file.
  
Limitations & Assumptions:-
--------------------------------------
My implementation is based on some assumptions. If I could get more time, I would have implemented relaxing those assumptions. Here I am listing the assumptions I made while implementing. 

1>I have added "#include"smc.h" at the end of all other includes. For this implementation, I needed to know the name of the last header file in the program. In all the test programs I used, helper_cuda.h was the last header file. So, getting the location of last header file, I added an extra header file after that. But I think if I would get more time, I could have implemented it without knowing the name of the last header file in the input program.
2>For detection of CUDA kernel function declaration, I used  CUDAGlobalAttr property. I am not sure whether all CUDA kernel function can be identified by this. 
3>In all the input programs which I used as test cases, blockIdx.x and blockIdx.y have been assigned to two separate variables. Looking into those examples, I thought that if I get the declaration of those two variables, then I can easily change the assigned value. So, I followed this approach and successfully replaced blockIdx.x and blockIdx.y as I was supposed to. But then I realized, there could be some cases where blockIdx.x and blockIdx.y have been used without being assigned. In those cases, my implementation would fail to rewrite these two variables.
I realized it too late to change my previous implementation. Then I tried to carefully look into the AST dump of input CUDA programs, I observed that blockIdx.x and blockIdx.y are not normal variables. They are GPU CUDA variables. So, they are structurally not like simple variables in the clang AST. If I could get more time, then I could implement this portion differently.




	


