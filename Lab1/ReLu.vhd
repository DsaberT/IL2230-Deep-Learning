library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;

Entity ReLU is
	port(
	input : IN sfixed((IntegerBit*2+1) downto -FractionalBit*2);
	output: OUT sfixed((IntegerBit*2+1) downto -FractionalBit*2));
end ReLU;

Architecture behave of ReLU is
  begin
    output <= input when (input > -1) else (others => '0');
  end behave;
