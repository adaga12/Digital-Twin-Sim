import { Flex, Heading, Separator, Table } from '@radix-ui/themes';
import { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { Link } from 'react-router-dom';
import { Routes } from './routes';

// Input data from the simulation
type AgentData = Record<string, number>;
type DataFrame = Record<string, AgentData>;
type DataPoint = [number, number, DataFrame];

// Output data to the plot
type PlottedAgentData = Record<string, number[]>;
type PlottedFrame = Record<string, PlottedAgentData>;

const App = () => {
  // Store plot data in state.
  const [positionData, setPositionData] = useState<PlottedAgentData[]>([]);
  const [velocityData, setVelocityData] = useState<PlottedAgentData[]>([]);
  const [initialState, setInitialState] = useState<DataFrame>({});

  useEffect(() => {
    // fetch plot data when the component mounts
    let canceled = false;

    async function fetchData() {
      console.log('calling fetchdata...');

      try {
        // data should be populated from a POST call to the simulation server
        const response = await fetch('http://localhost:8000/simulation');
        if (canceled) return;
        const data: DataPoint[] = await response.json();
        const updatedPositionData: PlottedFrame = {};
        const updatedVelocityData: PlottedFrame = {};

        setInitialState(data[0][2]);

        const baseData = () => ({
          x: [],
          y: [],
          z: [],
          type: 'scatter3d',
          mode: 'lines+markers',
          marker: { size: 4 },
          line: { width: 2 },
        });

        data.forEach(([t0, t1, frame]) => {
          for (let [agentId, { x, y, z, vx, vy, vz }] of Object.entries(frame)) {
            updatedPositionData[agentId] = updatedPositionData[agentId] || baseData();
            updatedPositionData[agentId].x.push(x);
            updatedPositionData[agentId].y.push(y);
            updatedPositionData[agentId].z.push(z);

            updatedVelocityData[agentId] = updatedVelocityData[agentId] || baseData();
            updatedVelocityData[agentId].x.push(vx);
            updatedVelocityData[agentId].y.push(vy);
            updatedVelocityData[agentId].z.push(vz);
          }
        });
        setPositionData(Object.values(updatedPositionData));
        setVelocityData(Object.values(updatedVelocityData));
        console.log('Set plot data!');
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    }

    fetchData();

    return () => {
      canceled = true;
    };
  }, []);

  return (
    <div
      style={{
        height: '100vh',
        width: '100vw',
        margin: '0 auto',
      }}
    >
      {/* Flex: https://www.radix-ui.com/themes/docs/components/flex */}
      <Flex direction="column" m="4" width="100%" justify="center" align="center">
        <Heading as="h1" size="8" weight="bold" mb="4">
          Simulation Data
        </Heading>
        <Link to={Routes.FORM}>Define new simulation parameters</Link>
        <Separator size="4" my="5" />
        <Flex direction="row" width="100%" justify="center">
          <Plot
            style={{ width: '45%', height: '100%', margin: '5px' }}
            data={positionData}
            layout={{
              title: 'Position',
              scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Z' },
              },
              autosize: true,
              dragmode: 'turntable',
            }}
            useResizeHandler
            config={{
              scrollZoom: true,
            }}
          />
          <Plot
            style={{ width: '45%', height: '100%', margin: '5px' }}
            data={velocityData}
            layout={{
              title: 'Velocity',
              scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Z' },
              },
              autosize: true,
              dragmode: 'turntable',
            }}
            useResizeHandler
            config={{
              scrollZoom: true,
            }}
          />
        </Flex>
        <Flex justify="center" width="100%" m="4">
          <Table.Root
            style={{
              width: '800px',
            }}
          >
            {/* Table: https://www.radix-ui.com/themes/docs/components/table */}
            <Table.Header>
              <Table.Row>
                <Table.ColumnHeaderCell>Agent</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Initial Position (x,y, z)</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Initial Velocity (x,y)</Table.ColumnHeaderCell>
              </Table.Row>
            </Table.Header>

            <Table.Body>
              {Object.entries(initialState).map(([agentId, { x, y, z, vx, vy, vz }]) => (
                <Table.Row key={agentId}>
                  <Table.RowHeaderCell>{agentId}</Table.RowHeaderCell>
                  <Table.Cell>
                    ({x}, {y}, {z})
                  </Table.Cell>
                  <Table.Cell>
                    ({vx}, {vy}, {vz})
                  </Table.Cell>
                </Table.Row>
              ))}
            </Table.Body>
          </Table.Root>
        </Flex>
      </Flex>
    </div>
  );
};

export default App;
