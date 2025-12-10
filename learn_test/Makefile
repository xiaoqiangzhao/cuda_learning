.PHONY: all

all: example
example:
	#nvcc example.cu -o example -keep --keep-dir inter_files
	# -arch specify GPU architecture for PTX and Cubin
	#  nvcc example.cu -arch=sm_<XY>/compute_<XY>/all
	#  -g debug info for host code 
	#  -G debug info for device code
	#  -lineinfo line-number infomation for device code
	#  -res-usage   print resource usage report after compilation, including registers, shared memory, constant memory and local memory for each kernel function
	#  -opt-info=inline: print information about inlined function
	#  -Xptxas=-warn-lmem-usage: Warn if local memory is used.
	#  -Xptxas=-warn-spills : Warn if registers are spilled to local memory
	#  -src-in-ptx: keep source code in PTX , need -lineinfo
	nvcc ../vectorAdd.cu -o example -keep --keep-dir inter_files -res-usage -Xptxas=-warn-spills  -lineinfo -src-in-ptx


clean:
	rm -rf example inter_files/*

show_info:
	nvcc --list-gpu-code
	nvcc --list-gpu-arch
