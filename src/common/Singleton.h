//
// Created by yons on 2021/6/15.
//

#ifndef VISIONENGINE_SINGLETON_H
#define VISIONENGINE_SINGLETON_H

namespace mirror {
    //! Generic singleton encapsulation structure
    template<class T> struct Singleton {
        //! Default constructor
        Singleton() : instance(nullptr) {}

        //! Destructor
        ~Singleton() { release(); }

        //! Releases the current instance
        inline void release() {
            if (instance) {
                delete instance;
                instance = nullptr;
            }
        }

        //! Current instance
        T *instance;
    };

}

#endif //VISIONENGINE_SINGLETON_H
