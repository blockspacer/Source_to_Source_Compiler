// This program implements the transformation of input cuda code to SM centric form

// Joymallya Chakraborty (jchakra@ncsu.edu)

#include<sstream>
#include <string>


#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"


using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;

 

static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

// Declaring three global bool variables to ensure single call of every functions
bool functionDeclFlag = true;
bool bxFlag = true;
bool byFlag = true;


// This class is for identification and rewriting of cuda kernel function declaration
class FunctionHandler : public MatchFinder::MatchCallback {
public:
	FunctionHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

	virtual void run(const MatchFinder::MatchResult &Result) {
	
		        const FunctionDecl *fd = Result.Nodes.getNodeAs<clang::FunctionDecl>("functiondecl");

	if(fd->hasAttr<CUDAGlobalAttr>() && functionDeclFlag == true)
	{		
	       		Stmt *FuncBody = fd->getBody();
	     	        QualType QT = fd->getReturnType();
	      	        std::string TypeStr = QT.getAsString();

	//	        Function name
	                DeclarationName DeclName = fd->getNameInfo().getName();
	                std::string FuncName = DeclName.getAsString();

	//              Add comment at the end
	                std::stringstream SSEnd;
	               	SSEnd << "\n  __SMC_End \n" ;		       
	                SourceLocation ST = fd->getSourceRange().getEnd();
	      	        Rewrite.InsertText(ST, SSEnd.str(), true, true);
	     	       
	//	        Add comment at the start			
	                std::stringstream SSBegin;
	                SSBegin << "\n  __SMC_Begin  "  ;
		        ST = FuncBody->getLocStart().getLocWithOffset(1);
 		        Rewrite.InsertText(ST, SSBegin.str(), true, true);
			

	//              Here I added some arguments to the argument list of GPU Kernel function 
			
			int numberOfParams = fd->getNumParams();	
			const ParmVarDecl * pv = fd->getParamDecl(numberOfParams-1); // --> getting the last parameter
				
		        Rewrite.InsertTextAfterToken(pv->getLocEnd(),", dim3 __SMC_orgGridDim, int __SMC_workersNeeded , int * __SMC_workerCount , int * __SMC_newChunkSeq, int * __SMC_seqEnds ");

	//	        making the bool value false
			functionDeclFlag = false;
	}
			
	}
private:
	Rewriter &Rewrite;
};

// This class is for identification and rewriting of cuda kernel call
class CallHandler : public MatchFinder::MatchCallback{
	public:
		CallHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

		virtual void run(const MatchFinder :: MatchResult &Result){

		const CUDAKernelCallExpr *cl = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>("functioncall");	
	
	// 	Introducing _SMC_init() before GPU Kernel call
		
		Rewrite.InsertText(cl->getLocStart(),"__SMC_init();\n\n",true,true);

	//      This part is for adding some arguments to GPU Kernel function call
	
		Rewrite.InsertText(cl->getLocEnd(),",__SMC_orgGridDim,__workersNeeded, __SMC_workesCount,__SMC_newChunkSeq, __SMC_seqEnds",true,true);
	}


private:	
  Rewriter &Rewrite;
};

// This class is for replacement of blockIdx.x
class BxReplacementHandler :public MatchFinder::MatchCallback{
	public:
		BxReplacementHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

		virtual void run(const MatchFinder :: MatchResult &Result){

		const VarDecl *bx = Result.Nodes.getNodeAs<clang::VarDecl>("bxvar");
		
		if(bxFlag == true){

			const Expr *bxvalue = bx->getInit();

			Rewrite.ReplaceText(bxvalue->getLocStart(),10,"(int)fmodf((float)__SMC_chunkID,(float)__SMC_orgGridDim.x)");// --> Replaces blockIdx.x

			bxFlag = false;
		}
		}
private:
	Rewriter &Rewrite;
};	

//This class is for replacement of blockIdx.y
class ByReplacementHandler :public MatchFinder::MatchCallback{
	public:
		ByReplacementHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

		virtual void run(const MatchFinder :: MatchResult &Result){
		
		const VarDecl *by = Result.Nodes.getNodeAs<clang::VarDecl>("byvar");
		
		if(byFlag == true){

		const Expr *byvalue = by->getInit();

	        Rewrite.ReplaceText(byvalue->getLocStart(),10,"(int)(__SMC_chunkID/__SMC_orgGridDim.x)"); // --> Replaces blockIdx.y
	        
		byFlag = false;

		}
		
	}
private:
	Rewriter &Rewrite;
};

// This class is for replacing grid call
class GridReplacementHandler : public MatchFinder::MatchCallback{
	public:
		GridReplacementHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

		virtual void run(const MatchFinder :: MatchResult &Result ){
		
	//      Replacing grid by __SMC_orgGridDim

		const VarDecl *gridcall = Result.Nodes.getNodeAs<clang::VarDecl>("gridcall");

	        const Expr *gridinit = gridcall->getInit();

	        Rewrite.ReplaceText(gridinit->getLocStart(),4,"__SMC_orgGridDim"); 

	}
private:
	Rewriter &Rewrite;
};

// This class is for adding smc.h header file at the end of all other includes
class Find_Includes : public PPCallbacks
{
	public:
		Find_Includes(Rewriter &Rewrite) : Rewrite(Rewrite) {}  
		

		    void InclusionDirective( SourceLocation hash_loc, const Token &include_token, StringRef file_name, bool is_angled, CharSourceRange filename_range,
							    const FileEntry *file, StringRef search_path, StringRef relative_path, const Module *imported)
			      {
				      // Add the new header file
				      
				     //llvm::outs() << file_name << "\n" ;	      
				      
				      if (file_name.find("helper_cuda.h") != std::string::npos){

				      SourceLocation ST = hash_loc.getLocWithOffset(25);
				      Rewrite.InsertText(ST,"\n#include \"smc.h\" \n ", true,true);				  
				      
				      }

			      }
private:
	Rewriter &Rewrite;
 };

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser. It registers a number of matchers and runs them on
// the AST.
class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R) :  HandlerForFunction(R),HandlerForCall(R),HandlerForBx(R),HandlerForBy(R), HandlerForGridReplacement(R) {
    			  
    		// Matcher for cuda kernel function declaration
	          Matcher.addMatcher(functionDecl().bind("functiondecl"),&HandlerForFunction);
	  	// Matcher for cuda kernel call	  
		  Matcher.addMatcher(cudaKernelCallExpr().bind("functioncall"),&HandlerForCall);
		// Matcher for blockIdx.x
	 	  Matcher.addMatcher(varDecl(hasName("bx")).bind("bxvar"),&HandlerForBx);
		// Matcher for blockIdx.y
		  Matcher.addMatcher(varDecl(hasName("by")).bind("byvar"),&HandlerForBy);
  		// Matcher for grid call		  
		  Matcher.addMatcher(varDecl(hasName("grid")).bind("gridcall"),&HandlerForGridReplacement);

  }
	
  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Matcher.matchAST(Context);
  }

private:

  FunctionHandler HandlerForFunction;
  CallHandler HandlerForCall;
  BxReplacementHandler HandlerForBx;
  ByReplacementHandler HandlerForBy;
  GridReplacementHandler HandlerForGridReplacement;
  MatchFinder Matcher;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}

bool BeginSourceFileAction(CompilerInstance &CI) 
{
	      std::unique_ptr<Find_Includes> find_includes_callback(new Find_Includes(TheRewriter));

              Preprocessor &pp = CI.getPreprocessor();
	      pp.addPPCallbacks(std::move(find_includes_callback));
	      return true;
}

void EndSourceFileAction() override {
 // For generating new output file --> This part will be uncommented if we want to generate new output file.
   /* std::error_code error_code;
    llvm::raw_fd_ostream outFile("../../Desktop/SMC_Changed.cu", error_code, llvm::sys::fs::F_None); // --> Proper path should be given 
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(outFile); // --> this will write the result to outFile
    outFile.close();*/   
   
//  This line will be commented if we want to generate new output file
  TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
      .write(llvm::outs()); //--> This prints output in the console
}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
						 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
