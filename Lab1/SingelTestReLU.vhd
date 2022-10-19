library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;

entity SingeltestReLU is
	port(
	all_A :IN A_file;
	all_B :IN B_file;
	clk   : IN std_logic;
	oup: OUT sfixed((IntegerBit*2+1) downto -FractionalBit*2));
end SingeltestReLU;

Architecture behave of SingeltestReLU is

	signal AL : sfixed(IntegerBit downto -FractionalBit) := (others=>'0');
	signal BL : sfixed(IntegerBit downto -FractionalBit) := (others=>'0');
	signal CL : sfixed((IntegerBit*2+1) downto -FractionalBit*2) := (others=>'0');
	signal OutL,inp : sfixed((IntegerBit*2+1) downto -FractionalBit*2);
	signal counter : unsigned(GenericN-1 downto 0) := (others=>'0');

begin	
	
	 ReLu:entity work.ReLU
			port map (
			input => inp,
			output => oup
			);
	
		MACR:entity work.MAC			  --Mac2
			port map (
			A => AL,
			B => BL,
			C => CL,
			clk => clk,
			Q => OutL);

	process(clk)
	begin
	if rising_edge(clk) then
		if counter > GenericN-1 then
		  inp <= OutL;
		else		
		AL <= all_A(to_integer(GenericN-1-counter));
		BL <= all_B(to_integer(GenericN-1-counter));
		CL <= OutL;
		counter <= counter+1;
		
		--AR <= all_A(GenericN/2+1+i);
		--BR <= all_B(GenericN/2+1+i);
		 --CR <= OutR; 
	      	end if;
	    end if;
	end process;
end behave;
