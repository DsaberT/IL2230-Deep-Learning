library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;

entity paralall is
	port(
	all_A :IN A_file;
	all_B :IN B_file;
	clk   : IN std_logic;
	--ResultSUM: OUT signed(2*BitLenght-1 downto 0));
	ResultSUM: OUT sfixed((IntegerBit*2+1) downto -FractionalBit*2));
end paralall;

Architecture behave of paralall is
	signal temp : result_type;

	begin	
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
			ResultSUM <= temp(GenericN);
		end if;
	end process;
end behave;
