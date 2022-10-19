library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_pkg.all;

package types_n_constan1 is
constant GenericN : integer := 3;
constant BitLenght : integer := 8;

constant FractionalBit : integer := 5;
constant IntegerBit : integer := BitLenght-FractionalBit-1;

type A_file is array (GenericN-1 downto 0) of sfixed(IntegerBit downto -FractionalBit);
type B_file  is array (GenericN-1 downto 0) of sfixed(IntegerBit downto -FractionalBit);
--not ufixed
type result_typeLeft is array (GenericN/2 downto 0) of signed (2*BitLenght-1  downto 0);
type result_typeRight is array (GenericN downto GenericN/2) of signed (2*BitLenght-1  downto 0);

type result_type is array (GenericN downto 0) of sfixed(IntegerBit*2+1 downto -FractionalBit*2);

end package;