set(LLVM_LINK_COMPONENTS support)

add_clang_executable(functions-rewrite
	FunctionRewrite.cpp
	      )
          target_link_libraries(functions-rewrite
		        clangTooling
			        clangBasic
				          clangASTMatchers
					              )

