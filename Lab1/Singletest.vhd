library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.fixed_pkg.all;

entity test is
	port(
	all_A :IN A_file;
	all_B :IN B_file;
	clk   : IN std_logic;
	--ResultSUM: OUT signed(2*BitLenght-1 downto 0));
	ResultSUM: OUT sfixed((IntegerBit*2+1) downto -FractionalBit*2));
end test;

Architecture behave of test is
	--signal ResultTempLeft :result_typeLeft;
        --signal ResultTempRight :result_typeRight;

	--signal AL :signed (BitLenght-1 downto 0):= (others=>'0');
	--signal BL :signed (BitLenght-1 downto 0):= (others=>'0');
	--signal CL :signed(2*BitLenght-1 downto 0):= (others=>'0');
	--signal OutL :signed(2*BitLenght-1 downto 0);
	--signal counter :unsigned(GenericN-1 downto 0) := (others=>'0');

	signal AL : sfixed(IntegerBit downto -FractionalBit) := (others=>'0');
	signal BL : sfixed(IntegerBit downto -FractionalBit) := (others=>'0');
	signal CL : sfixed((IntegerBit*2+1) downto -FractionalBit*2) := (others=>'0');
	signal OutL : sfixed((IntegerBit*2+1) downto -FractionalBit*2);
	signal counter : unsigned(GenericN-1 downto 0) := (others=>'0');

	--signal counter :unsigned(GenericN-1 downto 0) := (others=>'0');
	--signal AR :signed (BitLenght-1 downto 0);
	--signal BR :signed (BitLenght-1 downto 0);
	--signal CR :signed(2*BitLenght-1 downto 0);
	--signal OutR :signed(2*BitLenght-1 downto 0);
begin	
	
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
		  ResultSUM <= OutL;
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
