/*
 Authors
 Martin Schlather, schlather@math.uni-mannheim.de

 Copyright (C) 2020 -- 2022  Martin Schlather, Alexander Freudenberg, Johannes Naegele

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, writne to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
*/

bool useMMA1Bit(snpcoding method);
void crossprod_mma1Bit(Uint *CGM, Uint snps, Uint individuals, double *ans, bool warp, bool naive, unsigned int shape, size_t tilesize, size_t n_streams);
SEXP matrix_start_mma1Bit(Uint snps, Uint individuals, SEXP G);