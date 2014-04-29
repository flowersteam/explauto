#include "imle.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <boost/random.hpp>
#include<boost/tokenizer.hpp>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

using namespace std;

#define NOISE_Z 0.02
#define NOISE_X 0.1
#define NPOINTS_TRAIN 2000
#define NPOINTS_QUERY 10

#ifndef M_PI
    #define M_PI       3.14159265358979323846  // Visual Studio was reported not to define M_PI, even when including cmath and defining _USE_MATH_DEFINES...
#endif

int main(int argc, char **argv)
{

    string line;
    ifstream myfile ("example.txt");
    boost::char_separator<char> sep(" ");
    double v=0.0;




    string dummy;

    // Random number generation
    boost::mt19937 rng;
    boost::uniform_01<Scal> uniform;
    boost::normal_distribution<Scal> normal;

    const int d = 7;    //Input dimension
    const int D = 3;    //Output dimension
    typedef imle<d,D> IMLE;

    //      IMLE Parameters
    IMLE::Param param;
    param.wPsi = 100.0;
    param.Psi0 = Eig<d,D>::X::Constant(0.01);
    param.sigma0 = 1.0;
    param.p0 = 0.3;
    param.multiValuedSignificance = 0.9;

    // IMLE object
    IMLE imleObj(param);

    // Input and output Eigen Vectors
    IMLE::Z z = IMLE::Z::Zero();
    IMLE::X x;


    cout << "\t\tIMLE short demonstration:" << endl;
    cout << "\t#1 - Training:" << endl;
    cout << "Press <return> to continue..." << endl;
    getline( cin, dummy );

    if (myfile.is_open())
    {
    while ( getline (myfile,line) )
    {
        int i=0;
        boost::tokenizer<boost::char_separator<char> > tok(line, sep);
        for(boost::tokenizer<boost::char_separator<char> >::iterator beg=tok.begin(); beg!=tok.end();++beg){
            v = boost::lexical_cast<double>(*beg);
            if (i<d)
            {
                z[i]=v;
            }
            else
            {
                x[i-d]=v;
            }
            i+=1;
        }        
        imleObj.update(z,x);
    }
    myfile.close();
    }
    else cout << "Unable to open file\n"; 

    //for(int k = 0; k < NPOINTS_TRAIN; k++)
    //{
        //z[0] += uniform(rng) * NOISE_Z;
        //x[0] = cos(z[0]) + NOISE_X * normal(rng);

        //imleObj.update(z,x);
        //cout << "(" << z[0] << "," << x[0] << ") ";

        //if( z[0] > 4*M_PI )
            //z[0] = 0.0;
        //else
            //z[0] += 0.1;
    //}

    cout << "\n\n\nAfter " << NPOINTS_TRAIN << " training points, IMLE has activated " << imleObj.getNumberOfModels() << " linear models." << endl;
    cout << "\t\tPsi = [" << imleObj.getPsi().transpose() << "], sigma = [" << imleObj.getSigma().transpose() << "]." << endl;

    cout << "\t#2 - Forward Prediction:" << endl;
    cout << "Generating forward predictions at random input locations" << endl;
    cout << "Press <return> to continue..." << endl;
    getline( cin, dummy );

    for(int k = 0; k < NPOINTS_QUERY; k++)
    {
        for(int i=0; i<d; i++)
            z[i] = uniform(rng) * M_PI - 0.5*M_PI;

        x = imleObj.predict(z);
        // Alternatively:
        //      imleObj.predict(z);
        //      x = getPrediction()[0];

        //cout << "\t cos(" << z[0] << ") = " << cos(z[0]) << endl;
        cout << "\tpred(" << z << ") = " << x <<endl; // << "  (error = " << cos(z[0])-x[0] << ")" << endl;
    }


    cout << "\t#2 - Inverse Prediction:" << endl;
    cout << "Generating inverse predictions at random output locations" << endl;
    cout << "Press <return> to continue..." << endl;
    getline( cin, dummy );

    for(int k = 0; k < NPOINTS_QUERY; k++)
    {
        x[0] = uniform(rng) * 2.0 - 1.0;
        Scal inv_x = acos(x[0]);

        imleObj.predictInverse(x);

        vector<Scal> invSol;
        int nSol = imleObj.getNumberOfSolutionsFound();
        for( int k = 0; k < nSol; k++ )
            invSol.push_back( imleObj.getPrediction()[k][0] );
        sort(invSol.begin(), invSol.end() );

        cout << "\tinvCos(" << x[0] << ") = ( " << inv_x << ", " << 2.0*M_PI-inv_x << ", " << 2.0*M_PI+inv_x << ", " << 4.0*M_PI-inv_x << " )" << endl;
        cout << "\t  pred(" << x[0] << ") = ( ";
        for( int k = 0; k < nSol; k++ )
            cout << invSol[k] << ", ";
        cout << ")" << endl;
    }
}

