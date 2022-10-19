library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.Constants_Package.all;
use ieee.fixed_pkg.all;

entity Semi is 
  port(clk :IN std_logic;
       w :IN weight_array;
       x :IN x_array;
       q :OUT sfixed(I*2 downto F*2)
      );
end Semi;

Architecture behave of Semi is
  
  component MAC
    generic(N : Integer := 3);
    port(a,b : IN sfixed(I downto F);
         c : IN sfixed((I+1)*2 downto F*2);
         clk : IN std_logic;
         q: OUT sfixed((I+1)*2 downto F*2)
         );
  end component;
  
  signal q_temp_K: output_array;
  signal q_temp_N: output_array;
  
  begin
 
    GEN_MAC_K: for i in 0 to (M/2)-1 generate --4: 2st 5: 2 st 3: 1st
      MAKKE_K : MAC port map
        (w(i),x(i),q_temp_K(i),clk,q_temp_K(i+1));
    end generate GEN_MAC_K;
    
    GEN_MAC_N: for i in 0 to ((M + 1/2)/2) generate --4: 2st 5: 3 st 3: 2st
      MAKKE_N : MAC port map
        (w((M/2)-1+i),x((M/2)-1+i),q_temp_N(i),clk,q_temp_N(i+1));
    end generate GEN_MAC_N;
    
    process(clk)
      begin
        q_temp_N(0) <= (others => '0');
        q_temp_K(0) <= (others => '0');
        if rising_edge(clk) then
          q <= q_temp_K((M/2)-1) + q_temp_N((M + M/2)/2);
      end if;
    end process;
      
  end behave;
