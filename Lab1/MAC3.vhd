library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;


entity MAC is
	port(A,B: IN sfixed(IntegerBit downto -FractionalBit);
	--C: IN signed(2*BitLenght-1 downto 0);
	--Q: OUT signed(2*BitLenght-1 downto 0);
	C: IN sfixed((IntegerBit*2+1) downto -FractionalBit*2);
	Q: OUT sfixed((IntegerBit*2+1) downto -FractionalBit*2);
	clk: IN std_logic);
end MAC;
Architecture behave of MAC is
	signal temp : sfixed((IntegerBit*2+2) downto -(FractionalBit*2));
begin
	temp <= A*B+C;
	Q<= temp((IntegerBit*2+1) downto -FractionalBit*2);
end behave;
