library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;

entity SemiP_TB is
end SemiP_TB;

Architecture behave of SemiP_TB is
  
  component SemiP is
	 port(
	   all_A :IN A_file;
	 	 all_B :IN B_file;
	   clk   : IN std_logic;
	   ResultSum: OUT sfixed((IntegerBit*2+1) downto -FractionalBit*2));
  end component;
  
  signal A	      : A_file;
  signal B	      : B_file;
  signal clk     : std_logic := '0';
  signal ResultSum	 :sfixed((IntegerBit*2+1) downto -FractionalBit*2);
  
  begin
    DUT: SemiP port map(all_A=>A,all_B=>B,ResultSum=>ResultSum,clk=>clk);
    clk <= not(clk) after 5 ns;
    A <= ("01100100","00101000","01000000");
    B <= ("01110000","01100001","01011100");

end behave;
