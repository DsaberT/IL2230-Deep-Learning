library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;

entity MAC_TB is
end MAC_TB;

Architecture behave of MAC_TB is
  signal A	      : A_file;
  signal B	      : B_file;
  signal clk          : std_logic := '0';
 -- signal QL	      :signed(2*BitLenght-1 downto 0);
  --signal QR 	      :signed(2*BitLenght-1 downto 0);
 --signal Output	      :signed(2*BitLenght-1 downto 0);
  signal Output       : sfixed((IntegerBit*2+1) downto -FractionalBit*2);
begin
  
  Serial : entity work.test
	port map(
	All_A	=> A,
	All_B	=> B,
	clk 	=> clk,	
	ResultSUM => Output);

clk <= not(clk) after 5 ns;
A <= ("00100000","00100111","01010000");
B <= ("00100000","00100000","01010110");

end behave;
