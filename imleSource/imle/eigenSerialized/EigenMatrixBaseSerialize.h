friend class boost::serialization::access;



//inline virtual void resize(int i, int j)
//{
//    std::cout << "Base resize called: ERROR!" << std::endl;
//}
//
// When the class Archive corresponds to an output archive, the
// & operator is defined similar to <<.  Likewise, when the class Archive
// is a type of input archive the & operator is defined similar to >>.
template<class Archive>
inline void serialize(Archive & ar, const unsigned int version)
{
    int r = this->rows();     // makes sense for saving only
    int c = this->cols();

//    ar & r;
//    ar & c;
//
//    this->resize(r,c);        // makes sense for loading only
//    for(int j = 0; j < c; j++)
//	    for(int i = 0; i < r; i++)
//            ar & this->operator()(i,j);

    ar & boost::serialization::make_nvp("rows",r);
    ar & boost::serialization::make_nvp("cols",c);

    this->resize(r,c);        // makes sense for loading only

//    Scalar *data_p = this->data();
//    ar & boost::serialization::make_nvp("data",data_p);

    for(int j = 0; j < c; j++)
	    for(int i = 0; i < r; i++)
	    {

            ar & boost::serialization::make_nvp("data",this->operator()(i,j));
	    }
}


//template<class Archive>
//inline void save(Archive & ar, const unsigned int version) const
//{
//    ar & rows();
//    ar & cols();

//    for(int i = 0; i < r; i++)
//        for(int j = 0; j < r; j++)
//            ar & operator()(i,j);
//}

//template<class Archive>
//inline void load(Archive & ar, const unsigned int version)
//{
//    int r,c;

//    ar & r;
//    ar & c;
//
//    resize(r,c);
//    for(int i = 0; i < r; i++)
//        for(int j = 0; j < r; j++)
//            ar & operator()(i,j);
//}

//BOOST_SERIALIZATION_SPLIT_MEMBER()


