# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cap/code/test/ceres_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cap/code/test/ceres_demo/build

# Include any dependencies generated for this target.
include CMakeFiles/curvefitting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/curvefitting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/curvefitting.dir/flags.make

CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.o: CMakeFiles/curvefitting.dir/flags.make
CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.o: ../src/curve_fitting.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cap/code/test/ceres_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.o -c /home/cap/code/test/ceres_demo/src/curve_fitting.cpp

CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cap/code/test/ceres_demo/src/curve_fitting.cpp > CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.i

CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cap/code/test/ceres_demo/src/curve_fitting.cpp -o CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.s

# Object files for target curvefitting
curvefitting_OBJECTS = \
"CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.o"

# External object files for target curvefitting
curvefitting_EXTERNAL_OBJECTS =

curvefitting: CMakeFiles/curvefitting.dir/src/curve_fitting.cpp.o
curvefitting: CMakeFiles/curvefitting.dir/build.make
curvefitting: /usr/local/lib/libceres.a
curvefitting: /usr/lib/x86_64-linux-gnu/libglog.so
curvefitting: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
curvefitting: /usr/lib/x86_64-linux-gnu/libspqr.so
curvefitting: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
curvefitting: /usr/lib/x86_64-linux-gnu/libtbb.so
curvefitting: /usr/lib/x86_64-linux-gnu/libcholmod.so
curvefitting: /usr/lib/x86_64-linux-gnu/libccolamd.so
curvefitting: /usr/lib/x86_64-linux-gnu/libcamd.so
curvefitting: /usr/lib/x86_64-linux-gnu/libcolamd.so
curvefitting: /usr/lib/x86_64-linux-gnu/libamd.so
curvefitting: /usr/lib/x86_64-linux-gnu/liblapack.so
curvefitting: /usr/lib/x86_64-linux-gnu/libblas.so
curvefitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
curvefitting: /usr/lib/x86_64-linux-gnu/librt.so
curvefitting: /usr/lib/x86_64-linux-gnu/libcxsparse.so
curvefitting: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
curvefitting: /usr/lib/x86_64-linux-gnu/libtbb.so
curvefitting: /usr/lib/x86_64-linux-gnu/libcholmod.so
curvefitting: /usr/lib/x86_64-linux-gnu/libccolamd.so
curvefitting: /usr/lib/x86_64-linux-gnu/libcamd.so
curvefitting: /usr/lib/x86_64-linux-gnu/libcolamd.so
curvefitting: /usr/lib/x86_64-linux-gnu/libamd.so
curvefitting: /usr/lib/x86_64-linux-gnu/liblapack.so
curvefitting: /usr/lib/x86_64-linux-gnu/libblas.so
curvefitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
curvefitting: /usr/lib/x86_64-linux-gnu/librt.so
curvefitting: /usr/lib/x86_64-linux-gnu/libcxsparse.so
curvefitting: CMakeFiles/curvefitting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cap/code/test/ceres_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable curvefitting"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/curvefitting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/curvefitting.dir/build: curvefitting

.PHONY : CMakeFiles/curvefitting.dir/build

CMakeFiles/curvefitting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/curvefitting.dir/cmake_clean.cmake
.PHONY : CMakeFiles/curvefitting.dir/clean

CMakeFiles/curvefitting.dir/depend:
	cd /home/cap/code/test/ceres_demo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cap/code/test/ceres_demo /home/cap/code/test/ceres_demo /home/cap/code/test/ceres_demo/build /home/cap/code/test/ceres_demo/build /home/cap/code/test/ceres_demo/build/CMakeFiles/curvefitting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/curvefitting.dir/depend
