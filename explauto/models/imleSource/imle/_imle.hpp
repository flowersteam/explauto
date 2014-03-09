//#include "imle.hpp"

#include <algorithm>
#include <fstream>

#include <boost/math/distributions/chi_squared.hpp>
#include <Eigen/LU>

#define EXPAND(var) #var "= " << var
#define EXPAND_N(var) #var "=\n" << var

#define IMLE_CRTD		"IMLE object created."
#define IMLE_FNSHD		"IMLE object finished."
#define STR_BFR_UPDT	"You must start IMLE before updating."
#define OPENERR			"IMLE: Could not open file "
#define USNG_DEF_PRM    "IMLE: Using default parameters."


/*
 * Constructors and destructors
 */
template< int d, int D, template<int,int> class _Expert>
imle<d,D,_Expert>::imle(Param const &prm, int pre_alloc)
{
    init( prm, pre_alloc );
}

template< int d, int D, template<int,int> class _Expert>
imle<d,D,_Expert>::imle(std::string const &filename, int pre_alloc)
{
    init( Param(), pre_alloc );

	if( !load(filename) )
	    message(USNG_DEF_PRM);
    else
        message("Loaded " + filename);
}

template< int d, int D, template<int,int> class _Expert>
imle<d,D,_Expert>::~imle()
{
    if( param.saveOnExit )
        save(param.defaultSave);

	message(IMLE_FNSHD);
}

/*
 * Initialization
 */

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::init(Param const &prm, int pre_alloc)
{
    experts.reserve( pre_alloc );

    reset(prm);
	message(IMLE_CRTD);
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::reset(Param const &prm)
{
    experts.clear();
    setParameters(prm);

    noise_to_go = 0;
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::setParameters(Param const &prm)
{
    param = prm;
    set_internal_parameters();
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::set_internal_parameters()
{
    // Number of experts
    M = experts.size();

    // Initial guess for hyperparameters
    if( M == 0 )
    {
        sigma = Z::Constant(param.sigma0);
        Psi = param.Psi0;
    }

    // Significance test level
    sig_level_noiseX = quantile(boost::math::chi_squared(D), 1.0 - param.p0);
    sig_level_noiseZ = quantile(boost::math::chi_squared(d), 1.0 - param.p0);
    sig_level_noiseZX = quantile(boost::math::chi_squared(D+d), 1.0 - param.p0);

    pNoiseModelX = exp(-0.5*sig_level_noiseX) / sqrt(Psi.prod());
    pNoiseModelZ = exp(-0.5*sig_level_noiseZ) / sqrt(sigma.prod());
    pNoiseModelZX = exp(-0.5*sig_level_noiseZX) / sqrt(Psi.prod() * sigma.prod());

//    sig_level_multi_test = quantile(boost::math::chi_squared(D*(D+1)/2), param.multiValuedSignificance);
}


/*
 * Save and load
 */
template< int d, int D, template<int,int> class _Expert>
bool imle<d,D,_Expert>::save(std::string const &filename)
{
    std::ofstream fs(filename.c_str());

	if( !fs.is_open() )
	{
		message(OPENERR + filename);
		return false;
	}
    boost::archive::text_oarchive archive(fs);

	archive << (*this);
    fs.close();

    return true;
}

template< int d, int D, template<int,int> class _Expert>
bool imle<d,D,_Expert>::load(std::string const &filename)
{
    std::ifstream fs(filename.c_str());
	if( !fs.is_open() )
	{
		message(OPENERR + filename);
		return false;
	}
    boost::archive::text_iarchive archive(fs);

    reset();
	archive >> (*this);
    fs.close();

    return true;
}

template< int d, int D, template<int,int> class _Expert>
template<class Archive>
void imle<d,D,_Expert>::serialize(Archive & ar, const unsigned int version)
{
	// Parameters
	ar & param;

    // Experts
    ar & experts;

    // Common Priors
    ar & sigma;
    ar & Psi;

    // Remaining parameters (only for loading)
    set_internal_parameters();
    ar & noise_to_go;
}


/*
 * UPDATE
 */

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::update(Z const &z, X const &x)
{
    if( createNewExpert(z,x) )
	{
	    if( noise_to_go > 0)
	    {
	        noise_to_go--;
            return;
	    }

        // Create new linear expert
        experts.push_back( Expert(z,x,this) );
        M = experts.size();
	}

 	e_step(z,x);
	m_step();

	noise_to_go = param.nOutliers;
}

template< int d, int D, template<int,int> class _Expert>
bool imle<d,D,_Expert>::createNewExpert(Z const &z, X const &x)
{
    if( M == 0)
        return true;

    Scal sum_zx = 0.0;
    for(typename Experts::iterator it=experts.begin(); it < experts.end(); it++)
        sum_zx += it->queryZX(z,x);

    return sum_zx < pNoiseModelZX / M;

//    Scal sum_zx = 0.0;
//    for(typename Experts::iterator it=experts.begin(); it < experts.end(); it++)
//    {
//        it->queryZX(z,x);
//        sum_zx += it->get_rbf_zx();
//    }
//
//   return sum_zx < exp(-0.5*sig_level_noiseZX)  / M;
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::e_step(Z const &z, X const &x)
{
    Scal sum_h = 0.0;
    for(typename Experts::iterator it=experts.begin(); it < experts.end(); it++)
        sum_h += it->queryH(z,x);

    if( sum_h == 0.0)
    {
        // Create new linear expert
        experts.push_back( Expert(z,x,this) );
        experts.back().e_step(z,x,1.0);
        M = experts.size();
        std::cout << "e_step: sum_h = 0.0! (Should not happen)" << std::endl;
    }
    else
        for(typename Experts::iterator it=experts.begin(); it < experts.end(); it++)
            it->e_step( z, x, it->get_h()/sum_h );
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::m_step()
{
    // Update parameters for each linear expert
    for(typename Experts::iterator it=experts.begin(); it < experts.end(); it++)
        it->m_step();

    if( param.sphericalSigma0 )
    {
        Scal sum_trace = 0.0;
        for(typename Experts::iterator it=experts.begin(); it < experts.end(); it++)
            sum_trace += it->invSigma.trace();

        // Update common inverse-Wishart prior on Sigma
//        Scal tmp = (M*param.wSigma - 1.0)*d/2.0 - param.wsigma - 1.0;
        Scal tmp = M*param.wSigma*d/2.0 - param.wsigma - 1.0;
        Scal sigma0 = (tmp + sqrt( tmp*tmp + 2.0*param.wSigma*param.wsigma*param.sigma0*sum_trace))/(param.wSigma * sum_trace);
        sigma = Z::Constant(sigma0);
    }
    else
    {
        Z sum_diag = Z::Zero();
        for(typename Experts::iterator it=experts.begin(); it < experts.end(); it++)
            sum_diag += it->invSigma.diagonal();

        // Update common inverse-Wishart prior on Sigma
//        Scal tmp = (M * param.wSigma - 2.0 * param.wsigma - 3.0 ) /2.0;
        Scal tmp = M*param.wSigma/2.0 - param.wsigma - 1.0;
        sigma = 2.0 * param.wSigma * param.wsigma * param.sigma0 * sum_diag;
        sigma.array() += tmp*tmp;
        sigma.array() = (sigma.array().sqrt() + tmp) / (param.wSigma * sum_diag.array());
    }

//    pNoiseModelX = exp(-0.5*sig_level_noiseX) / sqrt(Psi.prod());
    pNoiseModelZ = exp(-0.5*sig_level_noiseZ) / sqrt(sigma.prod());
    pNoiseModelZX = exp(-0.5*sig_level_noiseZX) / sqrt(Psi.prod() * sigma.prod());
}



/*
 * PREDICTION
 */
template< int d, int D, template<int,int> class _Expert>
typename imle<d,D,_Expert>::X const &imle<d,D,_Expert>::predict(Z const &z)
{
    // Clear predict data structure
    prediction.clear();
    predictionVar.clear();
    predictionWeight.clear();
    predictionJacobian.clear();

    typename Experts::iterator it;
    Scal sum_p_z = 0.0;
    X sumInvRj = X::Zero();
    X sumInvRjXj = X::Zero();

    // Get updated information for all linear experts
    for(it=experts.begin(); it < experts.end(); it++)
        sum_p_z += it->queryZ(z);

    //Noise Model:
    Scal sumW = sum_p_z + pNoiseModelZ;

    for(it=experts.begin(); it < experts.end(); it++)
    {
        X invRj = it->getPredXInvVar() / (it->getUncertFactorPredX() + sumW / it->get_p_z());
        sumInvRj += invRj;
        sumInvRjXj += invRj.asDiagonal() * it->getPredX();
    }

    singleSol = sumInvRjXj.cwiseQuotient(sumInvRj);
    prediction.push_back(singleSol);
    predictionVar.push_back(sumInvRj.cwiseInverse());
    predictionWeight.push_back(sum_p_z/sumW);
    if( param.computeJacobian )
    {
        XZ jacobian = XZ::Zero();
        for(it=experts.begin(); it < experts.end(); it++)
            jacobian += it->get_p_z() * it->Lambda;
        predictionJacobian.push_back(jacobian/sum_p_z);
    }
    nSolFound = 1;

    return singleSol;
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::predictStrongest(Z const &z)
{
    predictMultiple(z);

    Scal w = 0.0;
    int best = 0;
    for(int i=0; i < nSolFound; i++)
        if( predictionWeight[i] > w )
        {
            w = predictionWeight[i];
            best = i;
        }

    prediction[0] = prediction[best];
    predictionVar[0] = predictionVar[best];
    predictionWeight[0] = predictionWeight[best];
    if( param.computeJacobian )
        predictionJacobian[0] = predictionJacobian[best];
    nSolFound = 1;

    while( prediction.size() > 1 )
    {
        prediction.pop_back();
        predictionVar.pop_back();
        predictionWeight.pop_back();
        predictionJacobian.pop_back();
    }
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::predictMultiple(Z const &z)
{
    // Clear predict data structure
    prediction.clear();
    predictionVar.clear();
    predictionWeight.clear();
    predictionJacobian.clear();

    int nSol = 1;
    Eigen::Matrix<Scal, Eigen::Dynamic, 1, 0, MAX_NUMBER_OF_SOLUTIONS> sum_w  = Vec::Zero(nSol);
    Eigen::Matrix<Scal, Eigen::Dynamic, 1, 0, MAX_NUMBER_OF_SOLUTIONS> sum_wSq = Vec::Zero(nSol);
    Eigen::Matrix<Scal, Eigen::Dynamic, 1, 0, MAX_NUMBER_OF_SOLUTIONS> sum_xInvRx = Vec::Zero(nSol);
    Eigen::Matrix<Scal, D, Eigen::Dynamic, D == 1 ? Eigen::RowMajor : Eigen::ColMajor, D, MAX_NUMBER_OF_SOLUTIONS> sum_invRx = Mat::Zero(D,nSol);
    Eigen::Matrix<Scal, D, Eigen::Dynamic, D == 1 ? Eigen::RowMajor : Eigen::ColMajor, D, MAX_NUMBER_OF_SOLUTIONS> sum_invR = Mat::Zero(D,nSol);
    Eigen::Matrix<Scal, D, Eigen::Dynamic, D == 1 ? Eigen::RowMajor : Eigen::ColMajor, D, MAX_NUMBER_OF_SOLUTIONS> sol(D,nSol);
    Eigen::Matrix<Scal, Eigen::Dynamic, Eigen::Dynamic, 0, INIT_SIZE, MAX_NUMBER_OF_SOLUTIONS> p(M,nSol);

    Vec sNearest = Vec::Zero(M);
    Vec w(M);
    Vec xInvRx(M);
    Mat invRx(D,M);
    Mat invR(D,M);

    Scal sumW = 0.0;
    for( int j = 0; j < M; j++ )
        sumW += experts[j].queryZ(z);

    if(sumW == 0.0)  // Not likely, but...
    {
        X sumXj = X::Zero();
        for( int j = 0; j < M; j++ )
            sumXj += experts[j].getPredX();
        prediction.push_back(sumXj / M);
        predictionVar.push_back(X::Constant( std::numeric_limits<Scal>::infinity() ));
        predictionWeight.push_back(0.0);
        if( param.computeJacobian )
        {
            XZ jacobian = XZ::Zero();
            for( int j = 0; j < M; j++ )
                jacobian += experts[j].Lambda;
            predictionJacobian.push_back(jacobian/M);
        }
        return;
    }

//    if( param.predictWithOutlierModel )
//        sumW += pNoiseModelZ;
    sumW += pNoiseModelZ;

    for( int j = 0; j < M; j++ )
    {
        w(j) = experts[j].get_p_z() / sumW;
        invR.col(j) = experts[j].getPredXInvVar() / (experts[j].getUncertFactorPredX() + 1.0 / w(j));
        invRx.col(j) = invR.col(j).asDiagonal() * experts[j].getPredX();
        xInvRx(j) = experts[j].getPredX().dot( invRx.col(j) );
    }

    int sBad;
    while( true )
    {
        // Statistical Test
        for( int j = 0; j < M; j++ )
        {
            sum_w(sNearest(j)) += w(j);
            sum_wSq(sNearest(j)) += w(j) * w(j);
            sum_invR.col(sNearest(j)) += invR.col(j);
            sum_invRx.col(sNearest(j)) += invRx.col(j);
            sum_xInvRx(sNearest(j)) += xInvRx(j);
        }
        sol = sum_invRx.cwiseQuotient( sum_invR );

       // Statistical Test for the need for new solutions
        if( sum_w.minCoeff(&sBad) == 0.0)
        {
            // One solution colapsed...
            nSol--;
            sum_w(sBad) = sum_w(nSol);              sum_w.resize(nSol);
            sum_invR.col(sBad) = sol.col(nSol);     sum_invR.resize(Eigen::NoChange,nSol);
            sol.col(sBad) = sol.col(nSol);          sol.resize(Eigen::NoChange,nSol);
            break;
        }

        Scal p_value, max_p_value = 0.0;
        Scal T, dof;
        for(int j = 0; j < nSol; j++)
        {
            if( (T = sum_xInvRx(j) - sol.col(j).dot( sum_invRx.col(j) ) ) <= 0 )
                continue;
            dof = (sum_w(j)*sum_w(j) / sum_wSq(j) - 1.0) * D + 1.0;

            if( ( p_value = cdf(boost::math::chi_squared(dof), T) ) >= max_p_value )
            {
                max_p_value = p_value;
                sBad = j;
            }
        }

        if( (max_p_value <= param.multiValuedSignificance && nSol >= param.nSolMin) || nSol >= param.nSolMax ) // Found all valid solutions:
            break;

        // New solution needed: allocate space, init sums
        nSol++;
        sum_w       = Vec::Zero(nSol);
        sum_wSq     = Vec::Zero(nSol);
        sum_xInvRx  = Vec::Zero(nSol);
        sum_invR    = Mat::Zero(D,nSol);
        sum_invRx   = Mat::Zero(D,nSol);

        sol.resize(Eigen::NoChange,nSol);
        p.resize(Eigen::NoChange,nSol);

        // init new solution to a sensible value
        int idxMax = 0, idx2Max = 0;
        Scal pMax = 0.0, p2Max = 0.0;
        for( int j = 0; j < M; j++ )
            if( sNearest(j) == sBad )
                if( w(j) > pMax )
                {
                    idx2Max = idxMax;
                    p2Max = pMax;
                    idxMax = j;
                    pMax = w(j);
                }
                else if( w(j) > p2Max )
                {
                    idx2Max = j;
                    p2Max = w(j);
                }
        sol.col(sBad) = experts[idxMax].getPredX();
        sol.col(nSol-1) = experts[idx2Max].getPredX();

        // E-M iterations
        Scal pSum;
        X dist;
        for( int nIter = 0; nIter < param.iterMax; nIter++ )
        {
            // E-Step
            for( int j = 0; j < M; j++ )
            {
                for( int s = 0; s < nSol; s++ )
                {
                    dist.noalias() = experts[j].getPredX() - sol.col(s);
                    p(j,s) = exp(-0.5*dist.dot(invR.col(j).asDiagonal()*dist));
                }
                if( (pSum = p.row(j).sum()) == 0.0 )
                    p.row(j).array() += 1.0/nSol;
                else
                    p.row(j) /= p.row(j).sum();
            }
            // M-Step
            sol.noalias() = (invRx * p).cwiseQuotient( invR * p );
            /* Here I can easily implement k-means by hard assigning to most probable solution */
        }

        // Hard assign solutions
        for( int j = 0; j < M; j++ )
            p.row(j).maxCoeff(&sNearest(j));
    }

    if( param.computeJacobian )
    {
        for( int s = 0; s < nSol; s++ )
        {
            prediction.push_back(sol.col(s));
            predictionVar.push_back(sum_invR.col(s).cwiseInverse());
            predictionWeight.push_back(sum_w(s));
            predictionJacobian.push_back(XZ::Zero());
        }
        for( int j = 0; j < M; j++ )
            predictionJacobian[sNearest(j)] += w(j) * experts[j].Lambda;  // Can have problems with solution collapsing...
        for( int s = 0; s < nSol; s++ )
            predictionJacobian[s] /= sum_w[s];
    }
    else
        for( int s = 0; s < nSol; s++ )
        {
            prediction.push_back(sol.col(s));
            predictionVar.push_back(sum_invR.col(s).cwiseInverse());
            predictionWeight.push_back(sum_w(s));
        }

    nSolFound = nSol;
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::predictInverse(X const &x)
{
    // Clear predict data structure
    prediction.clear();
    predictionVar.clear();
    predictionWeight.clear();
    predictionJacobian.clear();

    int nSol = 1;
    Eigen::Matrix<Scal, Eigen::Dynamic, 1, 0, MAX_NUMBER_OF_SOLUTIONS> sum_w  = Vec::Zero(nSol);
    Eigen::Matrix<Scal, Eigen::Dynamic, 1, 0, MAX_NUMBER_OF_SOLUTIONS> sum_wSq = Vec::Zero(nSol);
    Eigen::Matrix<Scal, Eigen::Dynamic, 1, 0, MAX_NUMBER_OF_SOLUTIONS> sum_zInvRz = Vec::Zero(nSol);
    Eigen::Matrix<Scal, d, Eigen::Dynamic, d == 1 ? Eigen::RowMajor : Eigen::ColMajor, d, MAX_NUMBER_OF_SOLUTIONS> sum_invRz = Mat::Zero(d,nSol);
    ArrayZZ sum_invR; sum_invR.reserve(MAX_NUMBER_OF_SOLUTIONS); sum_invR.push_back(ZZ::Zero());  // Just 1 element, careful if nSol != 1...
    ArrayZZ sum_invRp; sum_invRp.reserve(MAX_NUMBER_OF_SOLUTIONS); sum_invRp.push_back(ZZ::Zero());  // Just 1 element, careful if nSol != 1...

    Eigen::Matrix<Scal, d, Eigen::Dynamic, d == 1 ? Eigen::RowMajor : Eigen::ColMajor, d, MAX_NUMBER_OF_SOLUTIONS> sol(d,nSol);
    ArrayZZ solVar; solVar.reserve(MAX_NUMBER_OF_SOLUTIONS); solVar.push_back(ZZ::Zero());  // Just 1 element, careful if nSol != 1...
    Eigen::Matrix<Scal, Eigen::Dynamic, Eigen::Dynamic, 0, INIT_SIZE, MAX_NUMBER_OF_SOLUTIONS> p(M,nSol);

    Vec sNearest = Vec::Zero(M);
    Vec w(M);
    Vec zInvRz(M);
    Mat invRz(d,M);
    ArrayZZ invR; invR.reserve(M);

    Scal sumW = 0.0;
    for( int j = 0; j < M; j++ )
        sumW += experts[j].queryX(x);

    if(sumW == 0.0)  // Not likely, but...
    {
        Z sumZj = Z::Zero();
        for( int j = 0; j < M; j++ )
            sumZj += experts[j].getPredZ();
        prediction.push_back(sumZj / M);
        predictionVar.push_back(ZZ::Constant( std::numeric_limits<Scal>::infinity() ));
        predictionWeight.push_back(0.0);
        return;
    }

//    if( param.predictWithOutlierModel )
//        sumW += pNoiseModelX;
    sumW += pNoiseModelX;

    for( int j = 0; j < M; j++ )
    {
        w(j) = experts[j].get_p_x() / sumW;
//        invR.col(j) = experts[j].getPredZInvVar() / (experts[j].getUncertFactorPredZ() + 1.0 / w(j));
        invR.push_back(experts[j].getPredZInvVar() * w(j) );        //NO UNCERTAINTY...
        invRz.col(j) = invR[j] * experts[j].getPredZ();
        zInvRz(j) = experts[j].getPredZ().dot( invRz.col(j) );
    }

    int sBad;
    while( true )
    {
        // Statistical Test
        for( int j = 0; j < M; j++ )
        {
            sum_w(sNearest(j)) += w(j);
            sum_wSq(sNearest(j)) += w(j) * w(j);
            sum_invR[sNearest(j)] += invR[j];
            sum_invRz.col(sNearest(j)) += invRz.col(j);
            sum_zInvRz(sNearest(j)) += zInvRz(j);
        }

        for( int s = 0; s < nSol; s++ )
        {
            solVar[s] = sum_invR[s].inverse();
            sol.col(s) = solVar[s] * sum_invRz.col(s);

        }

        // Statistical Test for the need for new solutions
        if( sum_w.minCoeff(&sBad) == 0.0)
        {
            // One solution colapsed...
            nSol--;
            sum_w(sBad) = sum_w(nSol);      sum_w.resize(nSol);
            sol.col(sBad) = sol.col(nSol);  sol.resize(Eigen::NoChange,nSol);
            solVar[sBad] = solVar[nSol];    solVar.pop_back();
            break;
        }

        Scal p_value, max_p_value = 0.0;
        Scal T, dof;
        for(int j = 0; j < nSol; j++)
        {
            if( (T = sum_zInvRz(j) - sol.col(j).dot( sum_invRz.col(j) ) ) <= 0 )
                continue;
            dof = (sum_w(j)*sum_w(j) / sum_wSq(j) - 1.0) * d + 1.0;

            if( ( p_value = cdf(boost::math::chi_squared(dof), T) ) >= max_p_value )
            {
                max_p_value = p_value;
                sBad = j;
            }
        }

        if( (max_p_value <= param.multiValuedSignificance && nSol >= param.nSolMin) || nSol >= param.nSolMax ) // Found all valid solutions:
            break;



        // New solution needed: allocate space, init sums
        nSol++;
        sum_w       = Vec::Zero(nSol);
        sum_wSq     = Vec::Zero(nSol);
        sum_zInvRz  = Vec::Zero(nSol);
        sum_invRz   = Mat::Zero(d,nSol);
        for( int s = 0; s < nSol-1; s++ )
            sum_invR[s].setZero();
        sum_invR.push_back(ZZ::Zero());
        sum_invRp.push_back(ZZ::Zero());

        sol.resize(Eigen::NoChange,nSol);
        solVar.push_back(ZZ::Zero());
        p.resize(Eigen::NoChange,nSol);

        // init new solution to a sensible value
        int idxMax = 0, idx2Max = 0;
        Scal pMax = 0.0, p2Max = 0.0;
        for( int j = 0; j < M; j++ )
            if( sNearest(j) == sBad )
                if( w(j) > pMax )
                {
                    idx2Max = idxMax;
                    p2Max = pMax;
                    idxMax = j;
                    pMax = w(j);
                }
                else if( w(j) > p2Max )
                {
                    idx2Max = j;
                    p2Max = w(j);
                }
        sol.col(sBad) = experts[idxMax].getPredZ();
        sol.col(nSol-1) = experts[idx2Max].getPredZ();

        // E-M iterations
        Scal pSum;
        Z dist;
        for( int nIter = 0; nIter < param.iterMax; nIter++ )
        {
            // E-Step
            for( int j = 0; j < M; j++ )
            {
                for( int s = 0; s < nSol; s++ )
                {
                    dist.noalias() = experts[j].getPredZ() - sol.col(s);
                    p(j,s) = exp(-0.5*dist.dot(invR[j]*dist));
                }
                if( (pSum = p.row(j).sum()) == 0.0 )
                    p.row(j).array() += 1.0/nSol;
                else
                    p.row(j) /= p.row(j).sum();

                for( int s = 0; s < nSol; s++ )
                    sum_invRp[s] += invR[j] * p(j,s);
            }
            // M-Step
            for( int s = 0; s < nSol; s++ )
                sol.col(s).noalias() = sum_invRp[s].inverse() * (invRz * p.col(s));
            /* Here I can easily implement k-means by hard assigning to most probable solution */

            // Clear structure
            for( int s = 0; s < nSol; s++ )
                sum_invRp[s].setZero();
        }

        // Hard assign solutions
        for( int j = 0; j < M; j++ )
            p.row(j).maxCoeff(&sNearest(j));
    }

    for( int s = 0; s < nSol; s++ )
    {
        prediction.push_back(sol.col(s));
        predictionVar.push_back(solVar[s]);
        predictionWeight.push_back(sum_w(s));
    }

    nSolFound = nSol;
}


/*
 * Get's
 */

template< int d, int D, template<int,int> class _Expert>
typename imle<d,D,_Expert>::Param const &imle<d,D,_Expert>::getParameters()
{
    return param;
}

template< int d, int D, template<int,int> class _Expert>
typename imle<d,D,_Expert>::Experts const &imle<d,D,_Expert>::getExperts()
{
    return experts;
}

template< int d, int D, template<int,int> class _Expert>
typename imle<d,D,_Expert>::Z const &imle<d,D,_Expert>::getSigma()
{
    return sigma;
}

template< int d, int D, template<int,int> class _Expert>
typename imle<d,D,_Expert>::X const &imle<d,D,_Expert>::getPsi()
{
    return Psi;
}

template< int d, int D, template<int,int> class _Expert>
std::string imle<d,D,_Expert>::getName()
{
    return "IMLE";
}

//template< int d, int D>
//std::string imle<d,D, class LinearExpert<d,D> >::getName()
//{
//    return "IMLE";
//}
//
//template< int d, int D>
//std::string imle<d,D,FastLinearExpert<d,D> >::getName()
//{
//    return "FastIMLE";
//}

template< int d, int D, template<int,int> class _Expert>
int imle<d,D,_Expert>::getNumberOfModels()
{
    return M;
}

template< int d, int D, template<int,int> class _Expert>
ArrayVec const &imle<d,D,_Expert>::getPrediction()
{
    return prediction;
}

template< int d, int D, template<int,int> class _Expert>
ArrayMat const &imle<d,D,_Expert>::getPredictionVar()
{
    return predictionVar;
}

template< int d, int D, template<int,int> class _Expert>
ArrayScal const &imle<d,D,_Expert>::getPredictionWeight()
{
    return predictionWeight;
}

template< int d, int D, template<int,int> class _Expert>
ArrayMat const &imle<d,D,_Expert>::getPredictionJacobian()
{
    return predictionJacobian;
}

template< int d, int D, template<int,int> class _Expert>
typename MixtureOfLinearModels<d,D>::LinearModels imle<d,D,_Expert>::getLinearModels()
{
    int M = getNumberOfModels();
    typename MixtureOfLinearModels<d,D>::LinearModels models( M );
    LinearModel<d,D> *modelP;

    for(int i = 0; i<M; i++)
    {
        modelP = &experts[i];
        models[i] = *modelP;
    }

    return models;
}


/*
 * Print's
 */
template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::message(std::string const &msg)
{
	std::cout << msg << std::endl;
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::paramDisplay(std::ostream &out) const
{
    out << "-------------------------- Parameters ----" << std::endl;
    out << param;
}

template< int d, int D, template<int,int> class _Expert>
void imle<d,D,_Expert>::modelDisplay(std::ostream &out) const
{
   for(int i = 0; i < M; i++)
    {
        out << "------ #" << i+1 << ":" << std::endl;
        experts[i].modelDisplay(out);
        out << "------------------------------------------" << std::endl;
    }

    out << "-----------------------    Common    ----" << std::endl;
    out << EXPAND( M ) << std::endl;
    out << EXPAND(sigma) << std::endl;
    out << EXPAND(Psi) << std::endl;

}

template< int d, int D, template<int,int> class _Expert>
std::ostream &operator<<(std::ostream &out, imle<d,D,_Expert> const &imle_obj)
{
    imle_obj.paramDisplay(out);
    imle_obj.modelDisplay(out);

    return out;
}

template< int d, int D >
std::ostream &operator<<(std::ostream &out, imleParam<d,D> const &param)
{
    out << EXPAND(d) << std::endl;
    out << EXPAND(D) << std::endl;

    out << EXPAND(param.alpha) << std::endl;

    out << EXPAND(param.Psi0) << std::endl;
    out << EXPAND(param.sigma0) << std::endl;

    out << EXPAND(param.wsigma) << std::endl;
    out << EXPAND(param.wSigma) << std::endl;
    out << EXPAND(param.wNu) << std::endl;
    out << EXPAND(param.wLambda) << std::endl;
    out << EXPAND(param.wPsi) << std::endl;
    out << EXPAND(param.sphericalSigma0) << std::endl;

    out << EXPAND(param.p0) << std::endl;
    out << EXPAND(param.nOutliers) << std::endl;

    out << EXPAND(param.multiValuedSignificance) << std::endl;
    out << EXPAND(param.nSolMin) << std::endl;
    out << EXPAND(param.nSolMax) << std::endl;
    out << EXPAND(param.iterMax) << std::endl;
    out << EXPAND(param.computeJacobian) << std::endl;

    out << EXPAND(param.defaultSave) << std::endl;
    out << EXPAND(param.saveOnExit) << std::endl;

    return out;
}

