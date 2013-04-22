################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../lfcm.cu \
../load.cu \
../timing_of.cu 

CU_DEPS += \
./lfcm.d \
./load.d \
./timing_of.d 

OBJS += \
./lfcm.o \
./load.o \
./timing_of.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -lineinfo -pg -O0 -gencode arch=compute_12,code=sm_12 -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_12,code=compute_12 -gencode arch=compute_12,code=sm_12 -gencode arch=compute_13,code=compute_13 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_35,code=compute_35 -lineinfo -pg  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


