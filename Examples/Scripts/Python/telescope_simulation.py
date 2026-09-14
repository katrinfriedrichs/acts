#!/usr/bin/env python3

from html import parser
from pathlib import Path

import argparse

import acts
import acts.examples
from acts.examples.simulation import (
    MomentumConfig,
    addParticleGun,
    EtaConfig,
    PhiConfig,
    ParticleConfig,
    addFatras,
    addGeant4,
)

from acts.examples import (
    WhiteBoard,
    AlgorithmContext,
    ProcessCode,
    CsvTrackingGeometryWriter,
    ObjTrackingGeometryWriter,
)

u = acts.UnitConstants

parser = argparse.ArgumentParser(
    description="Full chain with the TelescopeDetector and a particle gun."
)

parser.add_argument(
    "--particle-type",
    "-p",
    help="Particle type",
    type=str,
    default="muon",
    choices=["muon", "electron", "pion"],
)  # pion = charged pion
parser.add_argument(
    "--positions_layers",
    help="Positions of the layers in the telescope detector",
    type=float,
    nargs="+",
    default=[30, 60, 90, 120, 150, 180, 210, 240, 270],
)
parser.add_argument(
    "--stereos_angles_layers",
    help="Stereo angles of the layers in the telescope detector",
    type=float,
    nargs="+",
    default=[0, 0, 0, 0, 0, 0, 0, 0, 0],
)
parser.add_argument(
    "--gun-pt-range",
    nargs=2,
    help="Pt range of the particle gun (GeV)",
    type=float,
    default=[1.0 * u.GeV, 1.0 * u.GeV],
)
parser.add_argument(
    "--disabled_Interactions",
    help="Disable interactions in the simulation",
    type=str,
    default="",
)  # "bremsstrahlung", "ionization", emScattering, emPhotonConversion
parser.add_argument(
    "--fixed_BetheBloch",
    help="Use Bethe-Bloch calculation (Landau Distribution Parameters were fixed) for energy loss simulation",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--outputObj",
    help="Output the geometry in OBJ format",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--Gen3", help="Use Gen3 for the simulation", action="store_true", default=False
)

args = parser.parse_args()

### Added Section to set particle type based on user input
if args.particle_type == "muon":
    pdg_T = acts.PdgParticle.eMuon
elif args.particle_type == "electron":
    pdg_T = acts.PdgParticle.eElectron
elif args.particle_type == "pion":
    pdg_T = acts.PdgParticle.ePionPlus
else:
    raise ValueError(f"Unknown particle type: {args.particle_type}")


if "__main__" == __name__:
    ### TODO: Add argument boolean Gen3
    detector = acts.examples.TelescopeDetector(
        bounds=[200, 200],
        positions=args.positions_layers,
        stereos=args.stereos_angles_layers,
        binValue=2,
    )
    trackingGeometry = detector.trackingGeometry()

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    num_layers = len(args.positions_layers)
    if args.positions_layers[0] < 0:
        num_layers = num_layers - 1

    outputDir = Path.cwd() / "telescope_simulation" / "output" / f"{num_layers}_layers"
    if not outputDir.exists():
        outputDir.mkdir(parents=True, exist_ok=True)

    if args.outputObj:
        (outputDir / "obj").mkdir(parents=True, exist_ok=True)
        if args.Gen3:
            context = acts.GeometryContext.dangerouslyDefaultConstruct()
            vis = acts.ObjVisualization3D()
            trackingGeometry.visualize(vis, context)
            vis.write(outputDir / "obj" / "geometry.obj")
        else:
            wb = WhiteBoard(level=acts.logging.INFO)
            context = AlgorithmContext(0, 0, wb, 0)

            writer = ObjTrackingGeometryWriter(
                level=acts.logging.INFO, outputDir=outputDir / "obj"
            )
            writer.write(context, trackingGeometry)

    outputDir = outputDir / f"{args.particle_type}_pT_{args.gun_pt_range[0]}GeV"
    if not outputDir.exists():
        outputDir.mkdir(parents=True, exist_ok=True)

    if args.disabled_Interactions == "_only_Ionization":
        enableInteractions = True
        enable_only_ionization = True
    elif args.disabled_Interactions == "":
        enableInteractions = True
        enable_only_ionization = False
    else:
        enableInteractions = False
        enable_only_ionization = False

    if args.fixed_BetheBloch:
        args.disabled_Interactions = "_fixed_BetheBloch" + args.disabled_Interactions
    for geant, postfix in [(False, "fatras" + args.disabled_Interactions)]:
        # for geant, postfix in [(False, "fatras"), (True, "geant4")]:
        rnd = acts.examples.RandomNumbers(seed=42)

        s = acts.examples.Sequencer(events=1, numThreads=1, logLevel=acts.logging.INFO)

        addParticleGun(
            s,
            MomentumConfig(
                args.gun_pt_range[0] * u.GeV,
                args.gun_pt_range[1] * u.GeV,
                transverse=True,
            ),
            EtaConfig(-10.0, 10.0, True),  # True => uniform in eta
            PhiConfig(0.0, 360.0 * u.degree),
            ParticleConfig(100000, pdg_T, False),
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(
                    0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 1.0 * u.ns
                ),
            ),
            multiplicity=1,
            rnd=rnd,
            outputDirRoot=outputDir / postfix,
        )

        if geant:
            addGeant4(
                s,
                detector,
                trackingGeometry,
                field,
                rnd=rnd,
                outputDirRoot=outputDir / postfix,
                outputDirCsv=outputDir / postfix,
                logLevel=acts.logging.VERBOSE,
            )
        else:
            addFatras(
                s,
                trackingGeometry,
                field,
                rnd=rnd,
                enableInteractions=enableInteractions,
                enable_only_ionization=enable_only_ionization,
                outputDirRoot=outputDir / postfix,
            )

        s.run()
