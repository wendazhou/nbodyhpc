#pragma once


namespace wenda {

namespace kdtree {


/** Utility function to access underlying container of a STL container adapter.
 * This function is used to get around the access protections of the STL container adapters,
 * and expose the protected member c.
 * 
 * @param adapter The STL container adapter to access.
 * 
 */
template <class ContainerAdapter>
typename ContainerAdapter::container_type & get_container_from_adapter (ContainerAdapter &adapter)
{
    struct hack : ContainerAdapter {
        static typename ContainerAdapter::container_type & get (ContainerAdapter &a) {
            return a.*&hack::c;
        }
    };

    return hack::get(adapter);
}

}

}
