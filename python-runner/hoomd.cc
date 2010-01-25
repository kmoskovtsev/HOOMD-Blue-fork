/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

using namespace boost::filesystem;

#include <string>
#include <sstream>
#include <list>
#include <iostream>
using namespace std;

/*! \file hoomd.cc
    \brief Executable for running python scripts with the hoomd module builtin
*/

//! forward declaration of the inithoomd function that inits the python module from hoomd_module.cc
extern "C" void inithoomd();

//! bring in the find_hoomd_data_dir from the python module
string find_hoomd_data_dir();

//! A simple method for finding the hoomd_script python module
string find_hoomd_script()
    {
    path hoomd_data_dir(find_hoomd_data_dir());
    list<path> search_paths;
    search_paths.push_back(hoomd_data_dir / "bin/python-module");
    search_paths.push_back(hoomd_data_dir / "lib/python-module");
    search_paths.push_back(hoomd_data_dir / "lib/hoomd/python-module");
    search_paths.push_back(hoomd_data_dir / ".." / "python-module");
    search_paths.push_back(hoomd_data_dir / ".." / "lib/python-module");
    search_paths.push_back(hoomd_data_dir / ".." / "lib/hoomd/python-module");
    search_paths.push_back(hoomd_data_dir / ".." / ".." / "python-module");
    search_paths.push_back(hoomd_data_dir / ".." / ".." / "lib/python-module");
    search_paths.push_back(hoomd_data_dir / ".." / ".." / "lib/hoomd/python-module");
    
    list<path>::iterator cur_path;
    for (cur_path = search_paths.begin(); cur_path != search_paths.end(); ++cur_path)
        {
        if (exists(*cur_path / "hoomd_script" / "__init__.py"))
            return cur_path->native_file_string();
        }
    cerr << endl 
         << "***Error! HOOMD python-module directory not found. Check your HOOMD directory file structure near " 
         << hoomd_data_dir.string() << endl << endl;
    return "";
    }

//! Main function for the executable
/*! \param argc argument count
    \param argv arguments
    Loads up the hoomd python module and then passes control onto Py_Main
*/
int main(int argc, char **argv)
    {
    char module_name[] = "hoomd";
    PyImport_AppendInittab(module_name, &inithoomd);
    Py_Initialize();
    
    // Need to inject the hoomd module path into sys.path
    string hoomd_script_dir = find_hoomd_script();
    if (hoomd_script_dir != "")
        {
        string python_cmds("import sys\n");
        python_cmds += string("sys.path.append(r\"") + hoomd_script_dir + string("\")\n");
        PyRun_SimpleString(python_cmds.c_str());
        }
        
    int retval = Py_Main(argc, argv);
    
    // trying to clean up python's messy memory leaks
    Py_Finalize();
    return retval;
    }
#ifdef WIN32
#pragma warning( pop )
#endif
