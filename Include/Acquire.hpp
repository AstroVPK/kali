#ifndef ACQUIRE_HPP
#define ACQUIRE_HPP

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>
#include <boost/system/linux_error.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/detail/quoted_manip.hpp>

using namespace std;
using namespace boost::filesystem;

template<typename InType> void AcquireInput(ostream& Os, istream& Is, const string& Prompt, const string& FailString, InType& Result) {
	do {
		Os << Prompt.c_str();
		if (Is.fail()) {
			Is.clear();
			Is.ignore(numeric_limits<streamsize>::max(), '\n');
			}
		Is >> Result;
		if (Is.fail()) {
			Os << FailString.c_str();
			Is.ignore(numeric_limits<streamsize>::max(), '\n');
			}
		} while(Is.fail());
	}

template<typename InType> InType AcquireInput(ostream& Os, istream& Is, const string& Prompt, const string& FailString) {
	InType temp;
	AcquireInput(Os,Is,Prompt,FailString,temp);
	return temp;
	}

void AcquireDirectory(ostream& Os, istream& Is, const string& Prompt, const string& FailString, string& Result);

void AcquireFile(ostream& Os, istream& Is, const string& Prompt, const string& FailString, string& Result);

#endif
