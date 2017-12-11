************************************************************************
            TVAL3D -- 3D Extension of TVAL3
************************************************************************

Modified by Dongjin Kwon


Introduction
====================

   TVAL3D is a 3D extension of TVAL3: "TV optimization -- an ALternating
minimization ALgorithm for Augmented Lagrangian functions", which is a TV
based image reconstruction and denosing solver. TVAL3 can be downloaded
from:

http://www.caam.rice.edu/~optimization/L1/TVAL3/

TVAL3 aims at solving the ill-possed inverse problem: approximately 
recover image ubar from

                   f = A*ubar + omega,                              (1)

where ubar is the original signal/image, A is a measurement matrix, omega 
is addtive noise and f is the noisy observation of ubar. 

Given A and f, TVAL3 tries to recover ubar by solving TV regularization 
problems:

                     min_u 	TV(u).               		    (2)
		      s.t.  A*u = f-omega

          or         min_u      TV(u)+||Au-f||_2^2


How To Use  
====================

Firstly, addpath both of TVAL3 and TVAL3D. TVAL3D is called in following ways:

               [U, out] = TVAL3D(A,b,p,q,r,opts)
 	   or         U = TVAL3D(A,b,p,q,r,opts).

Notice*:   Users should be aware that all fields of opts are assigned by default 
	   values, which are chosen based on authors' research or experience. 
	   However, at least one field of opts (any one) must be assigned by users.
