library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_n_constan1.all;
use ieee.math_real.all;
use ieee.fixed_pkg.all;


entity SemiPReLu is
	port(
	all_A :IN A_file;
	all_B :IN B_file;
	clk   : IN std_logic;
	oup: OUT sfixed((IntegerBit*2+1) downto -FractionalBit*2));
end SemiPReLu;

Architecture behave of SemiPReLu is
  
 	signal AL,AR : sfixed(IntegerBit downto -FractionalBit) := (others=>'0');
	signal BL,BR : sfixed(IntegerBit downto -FractionalBit) := (others=>'0');
	signal CL,CR : sfixed((IntegerBit*2+1) downto -FractionalBit*2):= (others=>'0');
	signal OutL,OutR : sfixed((IntegerBit*2+1) downto -FractionalBit*2);
 	signal counter :unsigned(GenericN-1 downto 0) := (others=>'0');
	signal temp : sfixed((IntegerBit*2+2) downto -FractionalBit*2);
	signal inp: sfixed((IntegerBit*2+1) downto -FractionalBit*2);
			
  begin
    ReLU:entity work.ReLU
			port map (
			input => inp,
			output => oup
			);
    
		MACL:entity work.MAC			  --Mac1
			port map (
			A => AL,
			B => BL,
			C => CL,
			clk => clk,
			Q => OutL);
			
		MACR:entity work.MAC			  --Mac2
			port map (
			A => AR,
			B => BR,
			C => CR,
			clk => clk,
			Q => OutR);
			
			process(clk)
			  variable n,i: integer;
	      begin
	      n := (GenericN-1) mod 2;
	      
	      if rising_edge(clk) then
		      if counter = ((GenericN-1)/2) and n = 0 then
		        i := integer(floor(real(GenericN-1)/real(2)));
		        
			      AR<= all_A(to_integer(i-counter));
		        BR <= all_B(to_integer(i-counter));
		          
		        CR <= OutR;
		        counter <= counter+1;
		      elsif counter > ((GenericN-1)/2) then
			      temp <= OutL+OutR;
		      
		      else		
		        AL <= all_A(to_integer(GenericN-1-counter));
		        BL <= all_B(to_integer(GenericN-1-counter));
		        
		        i := integer(floor(real(GenericN-1)/real(2)));
		        AR <= all_A(to_integer(i-counter));
		        BR <= all_B(to_integer(i-counter));
		      
		        CL <= OutL;
		        CR <= OutR;
		        counter <= counter+1;
	        end if;  
	    end if;
	end process;
	inp <= temp((IntegerBit*2+1) downto -FractionalBit*2);
    
end behave;
