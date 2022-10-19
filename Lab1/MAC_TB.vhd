library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;

entity MAC_TB is
end MAC_TB;

Architecture behave of MAC_TB is
  signal A,B	      : sfixed(IntegerBit downto -FractionalBit);
  signal C	        : sfixed((IntegerBit*2+1) downto -FractionalBit*2);
  signal clk          : std_logic := '0';
  signal Output       : sfixed((IntegerBit*2+1) downto -FractionalBit*2);
begin
  
  MAC_1 : entity work.MAC
	port map(
	A	=> A,
	B	=> B,
	C 	=> C,
	clk =>clk,
	Q => Output);

clk <= not(clk) after 5 ns;
A <= ("00100000");
B <= ("00100000");
C <=("0010000000100000");

end behave;
