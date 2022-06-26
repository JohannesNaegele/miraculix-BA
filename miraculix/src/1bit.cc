/*
 Authors 
 Martin Schlather, schlather@math.uni-mannheim.de


 Copyright (C) 2018 -- 2019  Martin Schlather

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.  
*/

#define MY_METHOD OneBit

#include "1bit.h"
#include "Bit23.intern.h"
#include "haplogeno.h"
#include "xport_import.h"
#include "align.h"
#include <time.h>
#include <omp.h>

#define UPI UnitsPerIndiv256_1

// start_snp should be zero; otherwise two consecutive matrices in this implementation cause bugs
void coding1(Uint *M, Uint start_individual, Uint end_individual, 
	       Uint start_snp, Uint end_snp, Uint Mnrow, SEXP Ans,
	       double VARIABLE_IS_NOT_USED *G) {
  if (start_snp % CodesPerUnit != 0) BUG; 
  Uint
    *info = GetInfo(Ans),
    cur_snps = end_snp, // deleted offset bc of 0 values
    *pM = M,
    // *endM = M + Mnrow * end_individual, // deleted offset bc of 0 values
    allUnits = Units(cur_snps), // half the size of 2-bit
    allUnitsM1 = allUnits - 1,
    rest = (cur_snps - allUnitsM1 * CodesPerUnit) * BitsPerCode,
    unitsPerIndiv = UPI(info[SNPS]), // half the size of 2-bit
    indivs = info[INDIVIDUALS],
    *code_1 = ((Uint*) INTEGER(Ans)), // Align256(Ans, ALIGN_HAPLOGENO)
    *code_2 = code_1 + unitsPerIndiv*indivs; // indivs ist Anzahl indivs
  // printf("unitsPerIndiv*start_individual: %u unitsPerIndiv*(start_individual + indivs): %u", unitsPerIndiv*start_individual, unitsPerIndiv*(start_individual + indivs));
  // PRINTF("Indivs %i\n", end_individual);
  // PRINTF("unitsPerIndiv %i\n", unitsPerIndiv);
  // PRINTF("BitsPerCode %i\n", BitsPerCode);
  Uint eins = 1;
  Uint zwei = 2;

  double t;
  #ifdef DO_PARALLEL
  t = omp_get_wtime();
  #endif

  // #ifdef DO_PARALLEL   
  // #pragma omp parallel for num_threads(CORES)
  // #endif
  for (Uint i = 0; i<(end_individual - start_individual); i++) {
    // printf("i: %u end_individual - start_individual: %u indivs: %u\n", i, end_individual - start_individual, indivs);
    Uint* code1 = code_1 + unitsPerIndiv*(i + start_individual);
    Uint* code2 = code_2 + unitsPerIndiv*(i + start_individual);
    Uint *mm = pM + Mnrow*i;
    for (Uint j=0; j<allUnits; j++) {
      Uint C1 = (Uint) 0;
      Uint C2 = (Uint) 0;
      Uint end = j == allUnitsM1 ? rest : BitsPerUnit;
      for (Uint shft = 0; shft < end; mm++, shft+=BitsPerCode) {
        C2 |= (*mm == zwei) << shft;
        C1 |= (*mm == eins) << shft;
      }
      code1[j] = C1;
      code2[j] = C2;
    }
  }
  t = omp_get_wtime() - t;
  // PRINTF("1-bit coding runs in %fs\n", t);
}