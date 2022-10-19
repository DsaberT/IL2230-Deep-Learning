library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;

entity paralallSig is
	port(
	all_A :IN A_file;
	all_B :IN B_file;
	clk   : IN std_logic;
	oup: OUT sfixed((IntegerBit*2+1) downto -FractionalBit*2));
end paralallSig;

Architecture behave of paralallSig is
	signal temp : result_type;
	signal inp, oup_temp : sfixed((IntegerBit*2+1) downto -FractionalBit*2);

	begin
	 Sig:entity work.Sigmoid
			port map (
			inp => inp,
			clk => clk,
			oup => oup_temp
			);  
	
	temp(0) <= (others => '0');
	gen_mac:
	FOR i IN 0 TO GenericN-1 GENERATE
		MACR:entity work.MAC			  --Mac2
			port map (
			A => all_A(i),
			B => all_B(i),
			C => temp(i),
			clk => clk,
			Q => temp(i+1));
	END GENERATE;

	process(clk)
	begin
		if rising_edge(clk) then
			inp <= temp(GenericN);
			oup <= oup_temp;
		end if;
	end process;
end behave;
