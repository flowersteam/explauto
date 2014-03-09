#ifndef __IMLE_H
#define __IMLE_H

#include "ILearner.hpp"
#include "EigenSerialized.h"
#include "expert.hpp"

#include <string>
#include <iostream>
#include <limits>

#define INIT_SIZE  1024
#define MAX_NUMBER_OF_SOLUTIONS 100
#define DEFAULT_SAVE "default.imle"

/*
 * imle Parameters
 */
template< int d, int D >
struct imleParam {
    // Parameters
	Scal alpha;

	typename Eig<d,D>::X Psi0;
	Scal sigma0;

	Scal wsigma;
	Scal wSigma;
	Scal wNu;
	Scal wLambda;
	Scal wPsi;
	bool sphericalSigma0;

	int nOutliers;
	Scal p0;

	Scal multiValuedSignificance;
//	bool predictWithOutlierModel;
	int nSolMin;
	int nSolMax;
	int iterMax;
	bool computeJacobian;

	// Saving
	bool saveOnExit;
	std::string defaultSave;

	imleParam()
	{
	    alpha = 0.99;

	    Psi0 = Eig<d,D>::X::Ones();
	    sigma0 = 1.0;

	    wsigma = 2*d;
        wSigma = pow(2.0,d);
        wNu = 0.0;
        wLambda = 0.1;
        wPsi = std::numeric_limits<Scal>::infinity();
        sphericalSigma0 = false;

	    nOutliers = 1;
	    p0 = 0.1;

	    multiValuedSignificance = 0.95;
//	    predictWithOutlierModel = true;
        nSolMin = 1;
        nSolMax = MAX_NUMBER_OF_SOLUTIONS;
        iterMax = 10;
        computeJacobian = false;

	    saveOnExit = false;
	    defaultSave = DEFAULT_SAVE;
    }

    // Boost serialization
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        // Dimensionality check
        int dd = d, DD = D;
        ar & BOOST_SERIALIZATION_NVP(dd);
        ar & BOOST_SERIALIZATION_NVP(DD);
        if( dd != d || DD != D )
        {
            std::cerr << "IMLE: Dimensions do not agree when loading file!\n";
            return;
        }

        ar & BOOST_SERIALIZATION_NVP(alpha);

        ar & BOOST_SERIALIZATION_NVP(Psi0);
        ar & BOOST_SERIALIZATION_NVP(sigma0);

        ar & BOOST_SERIALIZATION_NVP(wsigma);
        ar & BOOST_SERIALIZATION_NVP(wSigma);
        ar & BOOST_SERIALIZATION_NVP(wNu);
        ar & BOOST_SERIALIZATION_NVP(wLambda);
        Scal wPsiInv = 1.0/wPsi;  // XML serialization of infinity error...
        ar & BOOST_SERIALIZATION_NVP(wPsiInv);
        wPsi = 1.0 / wPsiInv;
        ar & BOOST_SERIALIZATION_NVP(sphericalSigma0);
//

        ar & BOOST_SERIALIZATION_NVP(nOutliers);
        ar & BOOST_SERIALIZATION_NVP(p0);

        ar & BOOST_SERIALIZATION_NVP(multiValuedSignificance);
        ar & BOOST_SERIALIZATION_NVP(nSolMin);
        ar & BOOST_SERIALIZATION_NVP(nSolMax);
        ar & BOOST_SERIALIZATION_NVP(iterMax);
        ar & BOOST_SERIALIZATION_NVP(computeJacobian);

        ar & BOOST_SERIALIZATION_NVP(saveOnExit);
        ar & BOOST_SERIALIZATION_NVP(defaultSave);
    }
};

template< int d, int D >
std::ostream &operator<<(std::ostream &out, imleParam<d,D> const &param);


template< int d, int D, template<int,int> class _Expert = ::FastLinearExpert  >
class imle : public MixtureOfLinearModels<d,D> //NonLinearRegressor<d,D>
{
public:
    typedef imleParam<d,D> Param;
    typedef _Expert<d,D> Expert;
    typedef std::vector< Expert, Eigen::aligned_allocator<Expert> > Experts;

    typedef typename Eig<d,D>::Z Z;
    typedef typename Eig<d,D>::X X;
    typedef typename Eig<d,D>::ZZ ZZ;
    typedef typename Eig<d,D>::XZ XZ;
    typedef typename Eig<d,D>::XX XX;

    typedef typename Eig<d,D>::ArrayZ ArrayZ;
    typedef typename Eig<d,D>::ArrayX ArrayX;
    typedef typename Eig<d,D>::ArrayZZ ArrayZZ;
    typedef typename Eig<d,D>::ArrayXX ArrayXX;
    typedef typename Eig<d,D>::ArrayXZ ArrayXZ;

	// Constructors and destructor
    imle(Param const &prm = Param(), int pre_alloc = INIT_SIZE);
	imle(std::string const &filename, int pre_alloc = INIT_SIZE);
	~imle();
	void reset()
    {   reset(param); }
	void reset(Param const &prm);

    // Save, load, init, params, display...
	void setParameters(Param const &prm);
	inline Param const &getParameters();

	bool save(std::string const &filename);
	bool load(std::string const &filename);

	inline Experts const &getExperts();
	inline Z const &getSigma();
	inline X const &getPsi();

    inline std::string getName();
    inline int getNumberOfModels();
    typename MixtureOfLinearModels<d,D>::LinearModels getLinearModels();

	void paramDisplay(std::ostream &out = std::cout) const;
	void modelDisplay(std::ostream &out = std::cout) const;


    // Algorithm
	void update(Z const &z, X const &x);

	X const & predict(Z const &z);               	// Single Forward Prediction
	void predictStrongest(Z const &z);          // Strongest Forward Prediction
	void predictMultiple(Z const &z);  	            // Multi Forward Prediction

//	void predictInverseSingle(X const &x);	    // Single Inverse Prediction
//	void predictInverseStrongest(X const &x);	// Strongest Inverse Prediction
	void predictInverse(X const &x);	            // Multiple Inverse Prediction

	inline ArrayVec  const &getPrediction();
	inline ArrayMat  const &getPredictionVar();
	inline ArrayScal const &getPredictionWeight();
	inline ArrayMat  const &getPredictionJacobian();
    inline int getNumberOfSolutionsFound() {return nSolFound; }

protected:
    // Save, load, init, params, display...
	void message(std::string const &msg);

	void init( Param const &prm, int pre_alloc );
	void set_internal_parameters();
    //      Boost serialization
    friend class boost::serialization::access;
    template<class Archive>
    inline void serialize(Archive & ar, const unsigned int version);

    // Algorithm
    bool createNewExpert(Z const &z, X const &x);
	void e_step(Z const &z, X const &x);
	void m_step();

	/*
	* CLASS DATA
	*/

	// Parameters
	Param param;

    // Experts
    Experts experts;
    int M;

    // Common priors
    Z sigma;
    X Psi;

    // Internal Parameters
	int noise_to_go;
	Scal sig_level_multi_test;
    Scal pNoiseModelZ, pNoiseModelX, pNoiseModelZX;
    Scal sig_level_noiseZ, sig_level_noiseX, sig_level_noiseZX;

	// Storage results
	int nSolFound;
	ArrayVec prediction;
	ArrayMat predictionVar;
	ArrayScal predictionWeight;
	ArrayMat predictionJacobian;
	X singleSol; // ...
};

template< int d, int D, template<int,int> class _Expert>
std::ostream &operator<<(std::ostream &out, imle<d,D,_Expert> const &imle_obj);



/*
 * typedefs
 */

// Only allowed in C++0x
//template< int d, int D>
//using IMLE = imle< d, D, LinearExpert<d,D> >;
//template< int d, int D>
//using FastIMLE = imle< d, D, FastLinearExpert<d,D> >;


// imle template implementation
#include "_imle.hpp"


#endif

