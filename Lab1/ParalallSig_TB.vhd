library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;
entity paralallMAC_TB is
end paralallMAC_TB;
Architecture behave of paralallMAC_TB is
signal A : A_file;
signal B : B_file;
signal clk : std_logic := '1';
-- signal QL :signed(2*BitLenght-1 downto 0);
--signal QR :signed(2*BitLenght-1 downto 0);
--signal Output :signed(2*BitLenght-1 downto 0);
signal Output : sfixed((IntegerBit*2+1) downto -FractionalBit*2);
begin
MAC_1 : entity work.paralallSig
port map(
all_A=> A,
all_B=> B,
clk => clk,
oup => Output);
clk <= not(clk) after 5 ns;
A <= ("00100000","00100111","01010000");
B <= ("00100000","00100000","01010110");
end behave;
